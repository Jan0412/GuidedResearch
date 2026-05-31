import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, Cout, D, H, W,
    Kd, Kh, Kw,
    S, P, Dil, G,
    Dout, Hout, Wout,
    stride_xb, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wc, stride_wi, stride_wd, stride_wh, stride_ww,
    stride_ob, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID mapping to output dimensions
    pid = tl.program_id(0)
    
    # Decompose pid into B, Cout, Dout, Hout, Wout
    # out_ptr shape: (B, Cout, Dout, Hout, Wout)
    ow = pid % Wout
    temp = pid // Wout
    oh = temp % Hout
    temp = temp // Hout
    od = temp % Dout
    temp = temp // Dout
    oc = temp % Cout
    b = temp // Cout

    # Group logic
    oc_per_group = Cout // G
    ic_per_group = Cin // G
    group_id = oc // oc_per_group
    ic_offset = group_id * ic_per_group

    # Accumulator for the convolution result
    acc = 0.0

    # Loop over the kernel spatial dimensions and input channels
    # Note: For high performance, these would be blocked, but for a general 
    # functional implementation, we iterate through the reduction dimensions.
    for ic in range(ic_per_group):
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Calculate input coordinates with stride, padding, and dilation
                    id_val = od * S + kd * Dil - P
                    ih_val = oh * S + kh * Dil - P
                    iw_val = ow * S + kw * Dil - P

                    # Boundary check for padding
                    if id_val >= 0 and id_val < D and ih_val >= 0 and ih_val < H and iw_val >= 0 and iw_val < W:
                        # Calculate pointers
                        # Input: x[b, ic_offset + ic, id_val, ih_val, iw_val]
                        x_off = (b * stride_xb + 
                                 (ic_offset + ic) * stride_xc + 
                                 id_val * stride_xd + 
                                 ih_val * stride_xh + 
                                 iw_val * stride_xw)
                        
                        # Weight: w[oc, ic, kd, kh, kw]
                        w_off = (oc * stride_wc + 
                                 ic * stride_wi + 
                                 kd * stride_wd + 
                                 kh * stride_wh + 
                                 kw * stride_ww)
                        
                        acc += tl.load(x_ptr + x_off) * tl.load(w_ptr + w_off)

    # Add bias if it exists
    if b_ptr is not None:
        acc += tl.load(b_ptr + oc)

    # Store result
    out_off = (b * stride_ob + 
               oc * stride_oc + 
               od * stride_od + 
               oh * stride_oh + 
               ow * stride_ow)
    tl.store(out_ptr + out_off, acc)

def triton_conv3d(x, weight, bias, stride, padding, dilation, groups):
    # Input shapes
    B, Cin, D, H, W = x.shape
    Cout, Cin_per_group, Kd, Kh, Kw = weight.shape
    
    # Output dimensions
    Dout = (D + 2 * padding - dilation * (Kd - 1) - 1) // stride + 1
    Hout = (H + 2 * padding - dilation * (Kh - 1) - 1) // stride + 1
    Wout = (W + 2 * padding - dilation * (Kw - 1) - 1) // stride + 1
    
    out = torch.empty((B, Cout, Dout, Hout, Wout), device=x.device, dtype=x.dtype)
    
    # Strides
    sx_b, sx_c, sx_d, sx_h, sx_w = x.stride()
    sw_c, sw_i, sw_d, sw_h, sw_w = weight.stride()
    so_b, so_c, so_d, so_h, so_w = out.stride()
    
    # Grid: one program per output element
    grid = (B * Cout * Dout * Hout * Wout,)
    
    conv3d_kernel[grid](
        x, weight, bias, out,
        B, Cin, Cout, D, H, W,
        Kd, Kh, Kw,
        stride, padding, dilation, groups,
        Dout, Hout, Wout,
        sx_b, sx_c, sx_d, sx_h, sx_w,
        sw_c, sw_i, sw_d, sw_h, sw_w,
        so_b, so_c, so_d, so_h, so_w,
        BLOCK_SIZE=1, # Scalar implementation for general correctness
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized 3D convolution using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We use nn.Conv3d to manage weights and bias initialization
        self.conv3d = nn.Conv3d(
            in_channels, 
            out_channels, 
            (kernel_size, kernel_size, kernel_size), 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous for the Triton kernel
        x = x.contiguous()
        weight = self.conv3d.weight.contiguous()
        bias = self.conv3d.bias.contiguous() if self.conv3d.bias is not None else None
        
        return triton_conv3d(
            x, 
            weight, 
            bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )