import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv1d_kernel(
    x_ptr,           # Input tensor (B, C, H, W)
    w_ptr,           # Weight tensor (C, 1, kernel_size)
    b_ptr,           # Bias tensor (C,)
    out_ptr,         # Output tensor (B, C, H, W_out)
    batch_size,      # B
    in_channels,     # C
    height,          # H
    width,           # W
    kernel_size,     # kernel_size (1D along width)
    stride,          # stride
    padding,         # padding
    dilation,        # dilation
    BLOCK_SIZE_W: tl.constexpr,  # Block size for width dimension
):
    # Program IDs: batch and channel are outer loops, width is computed
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Compute output width dimensions
    out_width = (width - (kernel_size - 1) * dilation - 1 + 2 * padding) // stride + 1
    
    # Offset to the current batch and channel in input
    x_offset = pid_b * in_channels * height * width + pid_c * height * width
    # Offset to the current channel in weight
    w_offset = pid_c * kernel_size
    
    # Compute output offset
    out_offset = pid_b * in_channels * height * out_width + pid_c * height * out_width
    
    # Load bias if available
    bias_val = 0.0
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + pid_c)
    
    # Process width dimension in blocks
    for pid_w in tl.range(0, out_width, BLOCK_SIZE_W):
        # Compute input width positions for this block of output positions
        w_out_start = pid_w * stride - padding
        offsets_w_in = w_out_start + tl.arange(0, BLOCK_SIZE_W)[None, :] * stride + tl.arange(0, kernel_size)[:, None] * dilation
        
        # Create mask for valid input width positions
        mask_in = (offsets_w_in >= 0) & (offsets_w_in < width)
        
        # Load input values: shape (kernel_size, BLOCK_SIZE_W)
        # We need to handle the fact that some positions might be out of bounds
        x_vals = tl.load(
            x_ptr + x_offset + tl.arange(0, height)[:, None, None] * width + offsets_w_in[None, :, :],
            mask=mask_in[None, :, :],
            other=0.0
        )
        
        # Load weights: shape (kernel_size,)
        w_vals = tl.load(w_ptr + w_offset + tl.arange(0, kernel_size))
        
        # Compute convolution: sum over kernel dimension
        # x_vals: (H, kernel_size, BLOCK_SIZE_W)
        # w_vals: (kernel_size,)
        # Result: (H, BLOCK_SIZE_W)
        conv_out = tl.sum(x_vals * w_vals[None, :, None], axis=1)
        
        # Add bias and store
        conv_out = conv_out + bias_val
        
        # Store output
        tl.store(
            out_ptr + out_offset + tl.arange(0, height)[:, None] * out_width + pid_w + tl.arange(0, BLOCK_SIZE_W)[None, :],
            conv_out,
            mask=(tl.arange(0, height)[:, None] < height) & (pid_w + tl.arange(0, BLOCK_SIZE_W)[None, :] < out_width)
        )


def triton_depthwise_conv1d(x, weight, bias, kernel_size, stride, padding, dilation):
    """
    Performs depthwise 1D convolution along width dimension using Triton.
    
    Args:
        x: Input tensor (B, C, H, W)
        weight: Weight tensor (C, 1, kernel_size)
        bias: Bias tensor (C,) or None
        kernel_size: Size of the kernel along width
        stride: Stride along width
        padding: Padding along width
        dilation: Dilation along width
    
    Returns:
        Output tensor (B, C, H, W_out)
    """
    batch_size, in_channels, height, width = x.shape
    
    # Compute output width
    out_width = (width - (kernel_size - 1) * dilation - 1 + 2 * padding) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, in_channels, height, out_width), dtype=x.dtype, device=x.device)
    
    # Set block size for width dimension
    BLOCK_SIZE_W = 64
    
    # Grid: (batch_size, in_channels)
    grid = (batch_size, in_channels)
    
    # Launch kernel
    depthwise_conv1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution with asymmetric kernel using Triton.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 1D convolution along width dimension.
        """
        return triton_depthwise_conv1d(
            x, self.weight, self.bias,
            self.kernel_size, self.stride, self.padding, self.dilation
        )