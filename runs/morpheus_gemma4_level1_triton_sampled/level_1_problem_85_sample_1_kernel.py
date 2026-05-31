import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, C, H, W,
    KH, KW,
    OH, OW,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    S_XB, S_XC, S_XH, S_XW,
    S_WC, S_W_C1, S_WKH, S_WKW,
    S_OB, S_OC, S_OOH, S_OOW,
    has_bias,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program ID for the (batch, channel, output_height) dimension
    pid_oh = tl.program_id(0)
    # Program ID for the output_width block
    pid_w = tl.program_id(1)

    # Decompose pid_oh into batch, channel, and output_height
    b = pid_oh // (C * OH)
    rem = pid_oh % (C * OH)
    c = rem // OH
    oh = rem % OH

    # Width offsets for the current block
    ow_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    mask_w = ow_offsets < OW

    # Accumulator for the convolution result
    acc = tl.zeros([BLOCK_SIZE_W], dtype=tl.float32)

    # Iterate over the kernel height and width
    for kh in range(KH):
        for kw in range(KW):
            # Calculate input coordinates
            ih = oh * stride_h + kh * dilation_h - padding_h
            iw = ow_offsets * stride_w + kw * dilation_w - padding_w
            
            # Boundary checks for the input tensor
            mask_h = (ih >= 0) & (ih < H)
            mask_iw = (iw >= 0) & (iw < W)
            full_mask = mask_w & mask_h & mask_iw
            
            # Load input value and weight value
            # Weight is (C, 1, KH, KW)
            x_val = tl.load(x_ptr + b * S_XB + c * S_XC + ih * S_XH + iw * S_XW, mask=full_mask, other=0.0)
            w_val = tl.load(w_ptr + c * S_WC + 0 * S_W_C1 + kh * S_WKH + kw * S_WKW)
            
            acc += x_val * w_val

    # Add bias if applicable
    if has_bias:
        bias_val = tl.load(bias_ptr + c)
        acc += bias_val

    # Store the result in the output tensor
    tl.store(out_ptr + b * S_OB + c * S_OC + oh * S_OOH + ow_offsets * S_OOW, acc, mask=mask_w)

def triton_depthwise_conv2d(x, weight, bias, stride, padding, dilation):
    # Input shapes
    B, C, H, W = x.shape
    KH, KW = weight.shape[2:]
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    # Output dimensions
    OH = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    OW = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1

    x = x.contiguous()
    weight = weight.contiguous()
    
    # Strides for input x (B, C, H, W)
    S_XB, S_XC, S_XH, S_XW = x.stride()
    # Strides for weight w (C, 1, KH, KW)
    S_WC, S_W_C1, S_WKH, S_WKW = weight.stride()
    
    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
    S_OB, S_OC, S_OOH, S_OOW = out.stride()

    has_bias = bias is not None
    bias_ptr = bias.contiguous() if has_bias else torch.zeros(1, device=x.device, dtype=x.dtype)

    BLOCK_SIZE_W = 64
    grid = (B * C * OH, triton.cdiv(OW, BLOCK_SIZE_W))

    depthwise_conv2d_kernel[grid](
        x, weight, bias_ptr, out,
        B, C, H, W,
        KH, KW,
        OH, OW,
        sh, sw,
        ph, pw,
        dh, dw,
        S_XB, S_XC, S_XH, S_XW,
        S_WC, S_W_C1, S_WKH, S_WKW,
        S_OB, S_OC, S_OOH, S_OOW,
        has_bias,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use nn.Conv2d to manage parameters (weight and bias)
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, (kernel_size_h, kernel_size_w), 
            stride=(stride_h, stride_w), padding=(padding_h, padding_w), 
            dilation=(dilation_h, dilation_w), groups=in_channels, bias=bias
        )
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )