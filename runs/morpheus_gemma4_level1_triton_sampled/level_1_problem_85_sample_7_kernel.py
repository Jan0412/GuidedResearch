import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C, H, W,
    KH, KW,
    SH, SW,
    PH, PW,
    DH, DW,
    OH, OW,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_c, w_stride_h, w_stride_w,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    has_bias,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Map pid_bc to batch and channel
    b = pid_bc // C
    c = pid_bc % C
    h_out = pid_h

    # Output width offsets
    w_start = pid_w * BLOCK_SIZE_W
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    out_mask = w_offsets < OW

    # Accumulator
    acc = tl.zeros([BLOCK_SIZE_W], dtype=tl.float32)

    # Convolution loops
    for kh in range(KH):
        for kw in range(KW):
            # Calculate input coordinates
            h_in = h_out * SH + kh * DH - PH
            w_in = w_offsets * SW + kw * DW - PW

            # Boundary masks for input
            # h_in is a scalar for this block of w_offsets
            h_mask = (h_in >= 0) & (h_in < H)
            w_mask = (w_in >= 0) & (w_in < W)
            load_mask = h_mask & w_mask & out_mask

            # Load input x and weight w
            # Weight is (C, 1, KH, KW), so we index by c, kh, kw
            x_off = b * x_stride_b + c * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
            w_off = c * w_stride_c + kh * w_stride_h + kw * w_stride_w
            
            val_x = tl.load(x_ptr + x_off, mask=load_mask, other=0.0)
            val_w = tl.load(weight_ptr + w_off)
            
            acc += val_x * val_w

    # Add bias
    if has_bias:
        bias_val = tl.load(bias_ptr + c)
        acc += bias_val

    # Store result
    out_off = b * out_stride_b + c * out_stride_c + h_out * out_stride_h + w_offsets * out_stride_w
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def triton_depthwise_conv2d(x, weight, bias, stride, padding, dilation):
    # Input shapes
    B, C, H, W = x.shape
    KH, KW = weight.shape[2], weight.shape[3]
    SH, SW = stride
    PH, PW = padding
    DH, DW = dilation

    # Output dimensions
    OH = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1

    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
    
    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
    w_stride_c, _, w_stride_h, w_stride_w = weight.stride()
    out_stride_b, out_stride_c, out_stride_h, out_stride_w = out.stride()

    has_bias = bias is not None
    bias_ptr = bias if has_bias else None

    BLOCK_SIZE_W = 64
    grid = (B * C, OH, (OW + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    depthwise_conv2d_kernel[grid](
        x, weight, bias_ptr, out,
        B, C, H, W,
        KH, KW,
        SH, SW,
        PH, PW,
        DH, DW,
        OH, OW,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        w_stride_c, w_stride_h, w_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        has_bias,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep the original Conv2d to maintain parameters and initialization
        self.conv2d = nn.Conv2d(
            in_channels, 
            in_channels, 
            (kernel_size_h, kernel_size_w), 
            stride=(stride_h, stride_w), 
            padding=(padding_h, padding_w), 
            dilation=(dilation_h, dilation_w), 
            groups=in_channels, 
            bias=bias
        )
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom Triton implementation instead of self.conv2d(x)
        return triton_depthwise_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )