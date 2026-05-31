import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wi, stride_wj,
    stride_on, stride_oc, stride_oh, stride_ow,
    S, P,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_W: tl.constexpr,
    has_bias: tl.constexpr,
):
    # Grid: (N * C, H_out, (W_out + BLOCK_W - 1) // BLOCK_W)
    pid_nc = tl.program_id(0)
    h_out = tl.program_id(1)
    pid_w = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C

    # Output width block
    w_out_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w_out = w_out_offsets < W_out

    # Base pointers for the current (n, c, h_out)
    out_ptr_base = out_ptr + n * stride_on + c * stride_oc + h_out * stride_oh
    x_ptr_base = x_ptr + n * stride_xn + c * stride_xc
    w_ptr_base = w_ptr + c * stride_wc

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    h_in_start = h_out * S - P

    for i in range(KH):
        h_in = h_in_start + i
        if h_in >= 0 and h_in < H:
            for j in range(KW):
                # Weight for (c, 0, i, j)
                weight = tl.load(w_ptr_base + i * stride_wi + j * stride_wj)
                
                # Input for (n, c, h_in, w_out * S - P + j)
                w_in_offsets = w_out_offsets * S - P + j
                mask_w_in = (w_in_offsets >= 0) & (w_in_offsets < W)
                
                x_vals = tl.load(x_ptr_base + h_in * stride_xh + w_in_offsets * stride_xw, 
                                 mask=mask_w_in, other=0.0)
                
                acc += x_vals * weight

    if has_bias:
        bias = tl.load(b_ptr + c)
        acc += bias

    tl.store(out_ptr_base + w_out_offsets * stride_ow, acc, mask=mask_w_out)

def triton_depthwise_conv2d(x, weight, bias, stride=1, padding=0):
    # x: (N, C, H, W)
    # weight: (C, 1, KH, KW)
    # bias: (C,) or None
    N, C, H, W = x.shape
    C_w, _, KH, KW = weight.shape
    
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1
    
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Strides
    stride_xn, stride_xc, stride_xh, stride_xw = C * H * W, H * W, W, 1
    stride_wc, stride_wi, stride_wj = KH * KW, KW, 1
    stride_on, stride_oc, stride_oh, stride_ow = C * H_out * W_out, H_out * W_out, W_out, 1

    BLOCK_W = 128
    grid = (N * C, H_out, (W_out + BLOCK_W - 1) // BLOCK_W)
    
    has_bias = bias is not None
    b_ptr = bias if has_bias else None

    depthwise_conv2d_kernel[grid](
        x, weight, b_ptr, out,
        N, C, H, W, H_out, W_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wc, stride_wi, stride_wj,
        stride_on, stride_oc, stride_oh, stride_ow,
        stride, padding,
        KH=KH, KW=KW,
        BLOCK_W=BLOCK_W,
        has_bias=has_bias,
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_enabled = bias
        
        # We use nn.Conv2d to manage parameters easily
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the Triton implementation instead of self.conv2d(x)
        return triton_depthwise_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            stride=self.stride, 
            padding=self.padding
        )