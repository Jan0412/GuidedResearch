import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,      # Input pointer
    w_ptr,      # Weight pointer
    b_ptr,      # Bias pointer
    out_ptr,    # Output pointer
    B, C, H, W, # Input dims
    OH, OW,     # Output dims
    S, P,       # Stride, Padding
    KH, KW,     # Kernel dims
    BLOCK_W: tl.constexpr,
):
    # Program ID for the batch and channel (B * C)
    pid_bc = tl.program_id(0)
    # Program ID for the output height (OH)
    pid_oh = tl.program_id(1)
    # Program ID for the output width block (OW // BLOCK_W)
    pid_ow = tl.program_id(2)

    # Decompose pid_bc into batch index and channel index
    b = pid_bc // C
    c = pid_bc % C

    # Current output height
    oh = pid_oh

    # Range of output widths for this block
    ow_offsets = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = ow_offsets < OW

    # Accumulator for the convolution
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Iterate over the kernel window
    for kh in range(KH):
        for kw in range(KW):
            # Calculate input coordinates
            ih = oh * S + kh - P
            iw = ow_offsets * S + kw - P
            
            # Boundary masks for padding
            mask_ih = (ih >= 0) & (ih < H)
            mask_iw = (iw >= 0) & (iw < W)
            mask = mask_ow & mask_ih & mask_iw

            # Load input value (B, C, H, W)
            # offset = b * (C * H * W) + c * (H * W) + ih * W + iw
            x_offset = b * C * H * W + c * H * W + ih * W + iw
            x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

            # Load weight value (C, 1, KH, KW)
            # offset = c * (KH * KW) + kh * KW + kw
            w_offset = c * KH * KW + kh * KW + kw
            w_val = tl.load(w_ptr + w_offset)

            acc += x_val * w_val

    # Load bias (C,)
    bias_val = tl.load(b_ptr + c)
    acc += bias_val

    # Store output (B, C, OH, OW)
    # offset = b * (C * OH * OW) + c * (OH * OW) + oh * OW + ow
    out_offset = b * C * OH * OW + c * OH * OW + oh * OW + ow_offsets
    tl.store(out_ptr + out_offset, acc, mask=mask_ow)


def triton_depthwise_conv2d(x, weight, bias, stride, padding):
    # Input shapes
    B, C, H, W = x.shape
    KH, KW = weight.shape[2], weight.shape[3]
    
    # Output shapes
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    
    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    
    if bias is None:
        bias = torch.zeros(C, device=x.device, dtype=x.dtype)
    else:
        bias = bias.contiguous()
        
    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
    
    BLOCK_W = 128
    grid = (B * C, OH, (OW + BLOCK_W - 1) // BLOCK_W)
    
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        B, C, H, W,
        OH, OW,
        stride, padding,
        KH, KW,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Keep the original Conv2d to manage parameters (weights and bias)
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom Triton kernel instead of the native PyTorch operator
        return triton_depthwise_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding
        )