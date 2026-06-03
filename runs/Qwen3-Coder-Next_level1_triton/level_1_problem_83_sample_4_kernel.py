import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv1d_kernel(
    x_ptr,              # Input tensor (B, C, H, W)
    w_ptr,              # Weight tensor (C, 1, kernel_size)
    b_ptr,              # Bias tensor (C,) - optional
    out_ptr,            # Output tensor (B, C, H, W_out)
    B, C, H, W,         # Input dimensions
    W_out,              # Output width
    kernel_size,        # Kernel size
    stride,             # Stride
    padding,            # Padding
    dilation,           # Dilation
    BLOCK_SIZE_W: tl.constexpr,  # Block size for parallelization over width
):
    # Program IDs: batch, channel, and height
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    height_idx = tl.program_id(2)
    
    # Calculate the starting position in the output width dimension
    output_w_start = tl.program_id(3) * BLOCK_SIZE_W
    
    # Load weight for this channel (only 1 filter per channel since groups=in_channels)
    # Weight shape: (C, 1, kernel_size), so we need weight[channel_idx, 0, :]
    w_offsets = tl.arange(0, kernel_size)
    weight = tl.load(w_ptr + channel_idx * kernel_size + w_offsets, 
                     mask=w_offsets < kernel_size, other=0.0)
    
    # Load bias if present
    bias_val = 0.0
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + channel_idx)
    
    # Process output width positions in this block
    output_w_offsets = output_w_start + tl.arange(0, BLOCK_SIZE_W)
    output_mask = output_w_offsets < W_out
    
    # Calculate the corresponding input positions for each output position
    # For convolution: input_pos = output_pos * stride - padding + kernel_pos * dilation
    input_w_offsets = output_w_offsets[:, None] * stride - padding + w_offsets[None, :] * dilation
    
    # Create masks for valid input positions
    valid_mask = (input_w_offsets >= 0) & (input_w_offsets < W)
    
    # Calculate the base offset for this batch and height
    batch_height_offset = (batch_idx * C * H + channel_idx * H + height_idx) * W
    
    # Compute the convolution
    acc = tl.zeros((BLOCK_SIZE_W,), dtype=tl.float32)
    
    for k in range(kernel_size):
        input_w = input_w_offsets[:, k]
        input_mask_k = output_mask & valid_mask[:, k]
        input_offset = batch_height_offset + input_w
        x_val = tl.load(x_ptr + input_offset, mask=input_mask_k, other=0.0)
        acc += x_val * weight[k]
    
    # Add bias and convert to output dtype
    out_val = acc + bias_val
    
    # Store result
    out_offset = (batch_idx * C * H * W_out + channel_idx * H * W_out + 
                  height_idx * W_out + output_w_offsets)
    tl.store(out_ptr + out_offset, out_val, mask=output_mask)


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Perform depthwise 2D convolution with asymmetric kernel (kernel_size, 1).
    
    Args:
        x: Input tensor (B, C, H, W)
        weight: Weight tensor (C, 1, kernel_size) - note: this matches nn.Conv2d with groups=C
        bias: Optional bias tensor (C,)
        stride, padding, dilation: Convolution parameters
    
    Returns:
        Output tensor (B, C, H, W_out)
    """
    B, C, H, W = x.shape
    kernel_size = weight.shape[2]
    
    # Calculate output dimensions
    W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, H, W_out), dtype=x.dtype, device=x.device)
    
    # Set up grid dimensions
    # Grid: (batch, channel, height, width_blocks)
    BLOCK_SIZE_W = 128
    width_blocks = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid = (B, C, H, width_blocks)
    
    # Launch kernel
    depthwise_conv1d_kernel[grid](
        x, weight, bias, out,
        B, C, H, W,
        W_out,
        kernel_size,
        stride, padding, dilation,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution with asymmetric kernel using Triton.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight tensor: (in_channels, 1, kernel_size)
        # Note: nn.Conv2d with groups=in_channels creates weights as (in_channels, 1, kernel_size, 1)
        # but for our Triton kernel we need (in_channels, 1, kernel_size)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the optimized Triton kernel.
        """
        return triton_depthwise_conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation
        )