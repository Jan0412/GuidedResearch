import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, D, H, W,
    C_out, Kd, Kh, Kw,
    D_out, H_out, W_out,
    S_x_b, S_x_c, S_x_d, S_x_h,
    S_w_oc, S_w_ic, S_w_kd, S_w_kh,
    S_out_b, S_out_oc, S_out_d, S_out_h,
    BLOCK_W: tl.constexpr,
):
    # Program ID mapping
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Decode pid0 to b, oc, d, h
    # pid0 = b * (C_out * D_out * H_out) + oc * (D_out * H_out) + d * H_out + h
    h = pid0 % H_out
    rem = pid0 // H_out
    d = rem % D_out
    rem = rem // D_out
    oc = rem % C_out
    b = rem // C_out

    # W dimension tiling
    w_offsets = pid1 * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w_offsets < W_out

    # Accumulator for the output elements in the current block
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels and kernel dimensions
    # Note: Since C_in, Kd, Kh, Kw are typically small, we use standard loops
    for ic in range(C_in):
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Calculate input pointer: x[b, ic, d + kd, h + kh, w_offsets + kw]
                    x_offset = (b * S_x_b + 
                                ic * S_x_c + 
                                (d + kd) * S_x_d + 
                                (h + kh) * S_x_h + 
                                w_offsets + kw)
                    x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                    
                    # Calculate weight pointer: w[oc, ic, kd, kh, kw]
                    w_offset = (oc * S_w_oc + 
                                ic * S_w_ic + 
                                kd * S_w_kd + 
                                kh * S_w_kh + 
                                kw)
                    w_val = tl.load(w_ptr + w_offset)
                    
                    acc += x_val * w_val

    # Add bias if it exists
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val

    # Store the result: out[b, oc, d, h, w_offsets]
    out_offset = (b * S_out_b + 
                  oc * S_out_oc + 
                  d * S_out_d + 
                  h * S_out_h + 
                  w_offsets)
    tl.store(out_ptr + out_offset, acc, mask=mask)

def triton_conv3d(x, weight, bias):
    # Input shapes
    B, C_in, D, H, W = x.shape
    C_out, _, Kd, Kh, Kw = weight.shape
    
    # Assuming stride=1, padding=0, dilation=1
    D_out = D - Kd + 1
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    
    out = torch.empty((B, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Strides for x
    S_x_b = C_in * D * H * W
    S_x_c = D * H * W
    S_x_d = H * W
    S_x_h = W
    
    # Strides for weight
    S_w_oc = C_in * Kd * Kh * Kw
    S_w_ic = Kd * Kh * Kw
    S_w_kd = Kh * Kw
    S_w_kh = Kw
    
    # Strides for output
    S_out_b = C_out * D_out * H_out * W_out
    S_out_oc = D_out * H_out * W_out
    S_out_d = H_out * W_out
    S_out_h = W_out
    
    BLOCK_W = 16
    grid = (B * C_out * D_out * H_out, (W_out + BLOCK_W - 1) // BLOCK_W)
    
    conv3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, D, H, W,
        C_out, Kd, Kh, Kw,
        D_out, H_out, W_out,
        S_x_b, S_x_c, S_x_d, S_x_h,
        S_w_oc, S_w_ic, S_w_kd, S_w_kh,
        S_out_b, S_out_oc, S_out_d, S_out_h,
        BLOCK_W=BLOCK_W
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the nn.Conv3d to handle parameter management (weights and bias)
        # This implementation focuses on stride=1, padding=0, dilation=1, groups=1
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous for Triton
        x = x.contiguous()
        weight = self.conv3d.weight.contiguous()
        bias = self.conv3d.bias.contiguous() if self.conv3d.bias is not None else None
        
        return triton_conv3d(x, weight, bias)