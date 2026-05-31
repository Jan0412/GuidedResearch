import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, H, W, Cout, Cin_pg, KH, KW,
    sh, sw, ph, pw, dh, dw, groups,
    sB, sC, sH, sW,
    sOC, sIC, sKH, sKW,
    sOB, sOC_out, sOH, sOW,
    BLOCK_IC: tl.constexpr,
):
    # Each program computes one output element (b, oc, oh, ow)
    pid = tl.program_id(0)
    
    # Decompose pid into output coordinates
    # We compute the total output size to map pid correctly
    # Note: Hout and Wout are passed implicitly via the grid calculation in the wrapper
    # But we need them here to decode pid if we use a 1D grid.
    # To be safer and simpler, we use a 3D grid (B*Cout, Hout, Wout).
    
    # For this implementation, we'll use a 3D grid:
    # pid_bc = tl.program_id(0)
    # pid_h = tl.program_id(1)
    # pid_w = tl.program_id(2)
    
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    b = pid_bc // Cout
    oc = pid_bc % Cout
    oh = pid_h
    ow = pid_w

    # Group logic: which input channels does this output channel look at?
    # oc_per_group = Cout // groups
    # group_id = oc // oc_per_group
    # ic_start = group_id * Cin_pg
    group_id = oc // (Cout // groups)
    ic_start = group_id * Cin_pg

    acc = 0.0

    # Loop over kernel spatial dimensions
    for kh in range(0, KH):
        for kw in range(0, KW):
            # Calculate input coordinates
            ih = oh * sh + kh * dh - ph
            iw = ow * sw + kw * dw - pw
            
            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                # Vectorize over input channels
                for ic_offset in range(0, Cin_pg, BLOCK_IC):
                    ic_offsets = ic_offset + tl.arange(0, BLOCK_IC)
                    mask = ic_offsets < Cin_pg
                    
                    # Load x: [B, Cin, H, W]
                    # Index: b*sB + (ic_start + ic_offsets)*sC + ih*sH + iw*sW
                    x_idx = b * sB + (ic_start + ic_offsets) * sC + ih * sH + iw * sW
                    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
                    
                    # Load w: [Cout, Cin_pg, KH, KW]
                    # Index: oc*sOC + ic_offsets*sIC + kh*sKH + kw*sKW
                    w_idx = oc * sOC + ic_offsets * sIC + kh * sKH + kw * sKW
                    w_val = tl.load(w_ptr + w_idx, mask=mask, other=0.0)
                    
                    acc += tl.sum(x_val * w_val)

    # Add bias
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val

    # Store result: [B, Cout, Hout, Wout]
    out_idx = b * sOB + oc * sOC_out + oh * sOH + ow * sOW
    tl.store(out_ptr + out_idx, acc)

def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    # Input shapes
    B, Cin, H, W = x.shape
    Cout, Cin_pg, KH, KW = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    # Calculate output dimensions
    Hout = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    Wout = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1

    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((B, Cout, Hout, Wout), device=x.device, dtype=x.dtype)

    # Get strides
    sB, sC, sH, sW = x.stride()
    sOC, sIC, sKH, sKW = weight.stride()
    sOB, sOC_out, sOH, sOW = out.stride()

    # Bias pointer
    b_ptr = bias if bias is not None else None

    # Grid: (B * Cout, Hout, Wout)
    grid = (B * Cout, Hout, Wout)
    
    # BLOCK_IC should be a power of 2
    BLOCK_IC = 32

    conv2d_kernel[grid](
        x, weight, b_ptr, out,
        B, Cin, H, W, Cout, Cin_pg, KH, KW,
        sh, sw, ph, pw, dh, dw, groups,
        sB, sC, sH, sW,
        sOC, sIC, sKH, sKW,
        sOB, sOC_out, sOH, sOW,
        BLOCK_IC=BLOCK_IC
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use nn.Conv2d to initialize weights and bias to maintain the same parameterization
        self.conv_params = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        # Store hyperparameters for the Triton kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the Triton implementation instead of the standard nn.Conv2d forward pass
        return triton_conv2d(
            x, 
            self.conv_params.weight, 
            self.conv_params.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )