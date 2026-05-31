import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, H, W,
    Cout, Cin_pg, KH, KW,
    sh, sw, ph, pw, dh, dw,
    G, Cout_pg, Hout, Wout,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Each program handles a block of the output width for a specific (batch, out_channel, out_height)
    pid_0 = tl.program_id(0)
    pid_w = tl.program_id(1)

    # Decompose pid_0 into batch, out_channel, and out_height
    # pid_0 = b * (Cout * Hout) + oc * Hout + oh
    B_Cout_Hout = Cout * Hout
    b = pid_0 // B_Cout_Hout
    rem = pid_0 % B_Cout_Hout
    oc = rem // Hout
    oh = rem % Hout

    # Output width offsets
    ow_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    mask_ow = ow_offsets < Wout

    # Accumulator for the convolution
    acc = tl.zeros((BLOCK_SIZE_W,), dtype=tl.float32)

    # Loop over input channels within the group
    for ic_pg in range(0, Cin_pg):
        # Calculate the actual input channel index based on groups
        ic = (oc // Cout_pg) * Cin_pg + ic_pg
        
        # Loop over kernel height and width
        for kh in range(0, KH):
            for kw in range(0, KW):
                # Calculate input coordinates
                ih = oh * sh + kh * dh - ph
                iw = ow_offsets * sw + kw * dw - pw
                
                # Boundary checks for input coordinates
                mask_ih = (ih >= 0) & (ih < H)
                mask_iw = (iw >= 0) & (iw < W)
                mask = mask_ow & mask_ih & mask_iw
                
                # Calculate pointers
                # x_ptr: [B, Cin, H, W]
                x_off = b * (Cin * H * W) + ic * (H * W) + ih * W + iw
                # w_ptr: [Cout, Cin_pg, KH, KW]
                w_off = oc * (Cin_pg * KH * KW) + ic_pg * (KH * KW) + kh * KW + kw
                
                # Load and multiply-accumulate
                x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0)
                w_val = tl.load(w_ptr + w_off)
                acc += x_val * w_val

    # Add bias if provided
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val

    # Store the result
    # out_ptr: [B, Cout, Hout, Wout]
    out_off = b * (Cout * Hout * Wout) + oc * (Hout * Wout) + oh * Wout + ow_offsets
    tl.store(out_ptr + out_off, acc, mask=mask_ow)


def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    B, Cin, H, W = x.shape
    Cout, Cin_pg, KH, KW = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    G = groups
    Cout_pg = Cout // G

    # Calculate output dimensions
    Hout = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    Wout = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    out = torch.empty((B, Cout, Hout, Wout), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_W = 16
    grid = (B * Cout * Hout, triton.cdiv(Wout, BLOCK_SIZE_W))
    
    conv2d_kernel[grid](
        x, weight, bias, out,
        B, Cin, H, W,
        Cout, Cin_pg, KH, KW,
        sh, sw, ph, pw, dh, dw,
        G, Cout_pg, Hout, Wout,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weight and bias as parameters
        # Weight shape: (out_channels, in_channels // groups, kH, kW)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )