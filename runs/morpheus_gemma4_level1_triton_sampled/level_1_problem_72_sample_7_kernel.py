import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, D, H, W,
    Cout, kD, kH, kW,
    sD, sH, sW,
    pD, pH, pW,
    Dout, Hout, Wout,
    Cin_pg, Cout_pg,
):
    # Each program computes one output element (b, oc, od, oh, ow)
    idx = tl.program_id(0)
    
    # Decompose flattened index
    ow = idx % Wout
    temp = idx // Wout
    oh = temp % Hout
    temp = temp // Hout
    od = temp % Dout
    temp = temp // Dout
    oc = temp % Cout
    b = temp // Cout

    # Group logic: find which group of input channels contributes to this output channel
    group_idx = oc // Cout_pg
    oc_local = oc % Cout_pg
    ic_start = group_idx * Cin_pg
    ic_end = (group_idx + 1) * Cin_pg

    acc = 0.0
    
    # Iterate over input channels in the group and the kernel dimensions
    # We use manual loops as the dimensions are small and vary per model instance
    for ic in range(ic_start, ic_end):
        for kd in range(kD):
            id_val = od + pD - kd
            if id_val >= 0 and id_val % sD == 0:
                id_idx = id_val // sD
                if id_idx < D:
                    for kh in range(kH):
                        ih_val = oh + pH - kh
                        if ih_val >= 0 and ih_val % sH == 0:
                            ih_idx = ih_val // sH
                            if ih_idx < H:
                                for kw in range(kW):
                                    iw_val = ow + pW - kw
                                    if iw_val >= 0 and iw_val % sW == 0:
                                        iw_idx = iw_val // sW
                                        if iw_idx < W:
                                            # Load input and weight
                                            # x: (B, Cin, D, H, W)
                                            x_off = b * (Cin * D * H * W) + ic * (D * H * W) + id_idx * (H * W) + ih_idx * W + iw_idx
                                            # w: (Cin, Cout_pg, kD, kH, kW)
                                            w_off = ic * (Cout_pg * kD * kH * kW) + oc_local * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw
                                            
                                            acc += tl.load(x_ptr + x_off) * tl.load(w_ptr + w_off)
    
    # Add bias if available
    if b_ptr is not None:
        acc += tl.load(b_ptr + oc)
        
    # Store the final result
    out_off = b * (Cout * Dout * Hout * Wout) + oc * (Dout * Hout * Wout) + od * (Hout * Wout) + oh * Wout + ow
    tl.store(out_ptr + out_off, acc)

def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    # Input shapes
    B, Cin, D, H, W = x.shape
    Cin_w, Cout_pg, kD, kH, kW = weight.shape
    Cout = Cout_pg * groups
    sD, sH, sW = stride
    pD, pH, pW = padding
    opD, opH, opW = output_padding
    
    # Calculate output dimensions
    Dout = (D - 1) * sD - 2 * pD + kD + opD
    Hout = (H - 1) * sH - 2 * pH + kH + opH
    Wout = (W - 1) * sW - 2 * pW + kW + opW
    
    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    out = torch.empty((B, Cout, Dout, Hout, Wout), device=x.device, dtype=x.dtype)
    
    Cin_pg = Cin // groups
    
    # Launch kernel
    num_elements = B * Cout * Dout * Hout * Wout
    grid = (num_elements,)
    
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, Cin, D, H, W,
        Cout, kD, kH, kW,
        sD, sH, sW,
        pD, pH, pW,
        Dout, Hout, Wout,
        Cin_pg, Cout_pg
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized version of the 3D transposed convolution using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use the standard layer to manage weights and bias initialization
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding, 
            groups=groups, bias=bias
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract params from the internal ConvTranspose3d layer
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else None
        
        # Use the custom Triton implementation
        return triton_conv_transpose3d(
            x, weight, bias, 
            self.stride, self.padding, self.output_padding, self.groups
        )