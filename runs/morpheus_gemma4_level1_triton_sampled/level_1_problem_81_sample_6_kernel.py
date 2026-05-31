import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, OC, HI, WI, KH, KW, HO, WO,
    S, P, D,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_ic, stride_w_oc, stride_w_kh, stride_w_kw,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_w = tl.program_id(0)
    pid_block = tl.program_id(1)

    # Map pid_w to batch, out_channel, and out_height
    b = pid_w // (OC * HO)
    rem = pid_w % (OC * HO)
    oc = rem // HO
    oh = rem % HO

    # Output width offsets for this block
    ow_offsets = pid_block * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = ow_offsets < WO

    # Accumulator for the output block
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Iterate over input channels and kernel dimensions
    # Using Python loops for small kernel/channel dimensions to generate Triton IR
    for ic in range(IC):
        for kh in range(KH):
            # Calculate input height index
            # Formula: oh = ih * S + kh * D - P  => ih = (oh + P - kh * D) / S
            ih_val = oh + P - kh * D
            if ih_val % S == 0:
                ih = ih_val // S
                if 0 <= ih < HI:
                    for kw in range(KW):
                        # Load weight value (scalar for the whole BLOCK_W)
                        w_ptr_off = ic * stride_w_ic + oc * stride_w_oc + kh * stride_w_kh + kw * stride_w_kw
                        weight_val = tl.load(w_ptr + w_ptr_off)

                        # Calculate input width indices for the block
                        # Formula: ow = iw * S + kw * D - P => iw = (ow + P - kw * D) / S
                        iw_val = ow_offsets + P - kw * D
                        mask_iw = (iw_val % S == 0) & (iw_val // S >= 0) & (iw_val // S < WI)
                        iw = iw_val // S

                        # Load input values
                        x_ptr_off = b * stride_x_b + ic * stride_x_c + ih * stride_x_h + iw * stride_x_w
                        val = tl.load(x_ptr + x_ptr_off, mask=mask_ow & mask_iw, other=0.0)
                        
                        acc += val * weight_val

    # Add bias if present
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val

    # Store result
    out_ptr_off = b * stride_out_b + oc * stride_out_c + oh * stride_out_h + ow_offsets * stride_out_w
    tl.store(out_ptr + out_ptr_off, acc, mask=mask_ow)


def triton_conv_transpose2d(x, weight, bias, stride, padding, dilation):
    # Tensors must be contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, IC, HI, WI = x.shape
    IC_w, OC, KH, KW = weight.shape
    
    # Output dimensions calculation
    HO = (HI - 1) * stride - 2 * padding + dilation * (KH - 1) + 1
    WO = (WI - 1) * stride - 2 * padding + dilation * (KW - 1) + 1
    
    out = torch.empty((B, OC, HO, WO), device=x.device, dtype=x.dtype)
    
    # Strides
    sx_b, sx_c, sx_h, sx_w = x.stride()
    sw_ic, sw_oc, sw_kh, sw_kw = weight.stride()
    so_b, so_c, so_h, so_w = out.stride()
    
    BLOCK_W = 16
    grid = (B * OC * HO, triton.cdiv(WO, BLOCK_W))
    
    b_ptr = bias if bias is not None else None

    conv_transpose2d_kernel[grid](
        x, weight, b_ptr, out,
        B, IC, OC, HI, WI, KH, KW, HO, WO,
        stride, padding, dilation,
        sx_b, sx_c, sx_h, sx_w,
        sw_ic, sw_oc, sw_kh, sw_kw,
        so_b, so_c, so_h, so_w,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weights and bias from the PyTorch layer
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias if self.conv_transpose2d.bias is not None else None
        
        return triton_conv_transpose2d(
            x, 
            weight, 
            bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )