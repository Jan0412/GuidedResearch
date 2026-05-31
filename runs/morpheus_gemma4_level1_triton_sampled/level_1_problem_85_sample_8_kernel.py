import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_channels, height, width,
    h_out, w_out,
    kh, kw,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_0 = tl.program_id(0)  # Maps to (batch * in_channels * h_out)
    pid_1 = tl.program_id(1)  # Maps to w_out blocks

    # Decompose pid_0
    # pid_0 = n * (in_channels * h_out) + c * h_out + h
    n = pid_0 // (in_channels * h_out)
    rem = pid_0 % (in_channels * h_out)
    c = rem // h_out
    h = rem % h_out

    # Output width offsets
    w_offsets = pid_1 * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    mask_w = w_offsets < w_out

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_W], dtype=tl.float32)

    # Iterate over the kernel dimensions
    # Since kh and kw are small, we loop over them
    for i in range(kh):
        h_in = h * stride_h + i * dilation_h - padding_h
        if h_in >= 0 and h_in < height:
            for j in range(kw):
                # Calculate input width indices
                w_in = w_offsets * stride_w + j * dilation_w - padding_w
                
                # Mask for width and boundary
                mask = mask_w & (w_in >= 0) & (w_in < width)
                
                # Load input: x[n, c, h_in, w_in]
                # x_ptr shape: (batch, in_channels, height, width)
                x_offset = n * (in_channels * height * width) + \
                           c * (height * width) + \
                           h_in * width + w_in
                val_x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                
                # Load weight: w[c, 0, i, j]
                # w_ptr shape: (in_channels, 1, kh, kw)
                w_offset = c * (kh * kw) + i * kw + j
                val_w = tl.load(w_ptr + w_offset)
                
                acc += val_x * val_w

    # Add bias if it exists
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + c)
        acc += bias_val

    # Store result: out[n, c, h, w_offsets]
    # out_ptr shape: (batch, in_channels, h_out, w_out)
    out_offset = n * (in_channels * h_out * w_out) + \
                 c * (h_out * w_out) + \
                 h * w_out + w_offsets
    tl.store(out_ptr + out_offset, acc, mask=mask_w)


def triton_depthwise_conv2d(x, weight, bias, stride, padding, dilation):
    # x: (N, C, H, W)
    # weight: (C, 1, Kh, Kw)
    # bias: (C,) or None
    batch, in_channels, height, width = x.shape
    kh, kw = weight.shape[2], weight.shape[3]
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    # Calculate output dimensions
    h_out = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    out = torch.empty((batch, in_channels, h_out, w_out), device=x.device, dtype=x.dtype)

    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    BLOCK_SIZE_W = 64
    grid = (batch * in_channels * h_out, (w_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch, in_channels, height, width,
        h_out, w_out,
        kh, kw,
        sh, sw,
        ph, pw,
        dh, dw,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # We keep the original nn.Conv2d to manage parameters (weight and bias)
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, (kernel_size_h, kernel_size_w), 
            stride=(stride_h, stride_w), padding=(padding_h, padding_w), 
            dilation=(dilation_h, dilation_w), groups=in_channels, bias=bias
        )
        
        # Store parameters for easy access in forward
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using the custom Triton kernel.
        """
        # Use the Triton implementation instead of the PyTorch one
        return triton_depthwise_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )