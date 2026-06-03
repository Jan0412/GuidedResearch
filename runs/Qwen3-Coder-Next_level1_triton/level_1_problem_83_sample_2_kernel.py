import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv1d_kernel(
    x_ptr,  # Input tensor pointer (B, C, H, W)
    w_ptr,  # Weight tensor pointer (C, 1, kernel_size)
    b_ptr,  # Bias tensor pointer (C,) or None
    out_ptr,  # Output tensor pointer (B, C, H, W_out)
    B: tl.constexpr,  # Batch size
    C: tl.constexpr,  # Number of channels
    H: tl.constexpr,  # Height
    W: tl.constexpr,  # Input width
    W_out: tl.constexpr,  # Output width
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles one (batch, channel, height) combination
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    height_idx = tl.program_id(2)
    
    # Calculate input pointer offset for this (batch, channel, height)
    input_offset = batch_idx * (C * H * W) + channel_idx * (H * W) + height_idx * W
    
    # Calculate output pointer offset for this (batch, channel, height)
    output_offset = batch_idx * (C * H * W_out) + channel_idx * (H * W_out) + height_idx * W_out
    
    # Load kernel weights for this channel (only one kernel per channel since groups=in_channels)
    # Kernel shape is (1, kernel_size) in the weight tensor, but stored as (C, 1, kernel_size)
    kernel_offsets = tl.arange(0, kernel_size)
    kernel_ptr = w_ptr + channel_idx * kernel_size + kernel_offsets
    kernel = tl.load(kernel_ptr, mask=kernel_offsets < kernel_size, other=0.0)
    
    # Process output width dimension in blocks
    num_blocks = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    for block_idx in range(num_blocks):
        # Calculate output width indices for this block
        out_w = block_idx * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
        mask = out_w < W_out
        
        # Calculate corresponding input positions with stride, padding, and dilation
        # Formula: input_pos = out_w * stride - padding + kernel_pos * dilation
        kernel_pos = tl.arange(0, kernel_size)
        input_w = out_w[:, None] * stride - padding + kernel_pos[None, :] * dilation
        
        # Create mask for valid input positions
        valid_mask = (input_w >= 0) & (input_w < W)
        
        # Load input values: shape (BLOCK_SIZE_W, kernel_size)
        input_offsets = input_w * 1 + input_offset  # Since we're only accessing width dimension
        input_values = tl.load(x_ptr + input_offsets[:, :, None], 
                              mask=valid_mask[:, :, None], 
                              other=0.0)
        
        # Reshape to (BLOCK_SIZE_W, kernel_size) for multiplication
        input_values = input_values[:, :, 0]  # Remove the last dimension
        
        # Compute convolution: sum over kernel dimension
        # Multiply input with kernel and sum
        conv_result = tl.sum(input_values * kernel[None, :], axis=1)
        
        # Add bias if present
        if HAS_BIAS:
            bias = tl.load(b_ptr + channel_idx)
            conv_result = conv_result + bias
        
        # Store result
        out_offsets = output_offset + out_w
        tl.store(out_ptr + out_offsets, conv_result, mask=mask)


def triton_depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0, dilation: int = 1):
    """
    Performs depthwise 1D convolution along width dimension using Triton.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Weight tensor of shape (C, 1, kernel_size) - note the shape
        bias: Optional bias tensor of shape (C,)
        stride, padding, dilation: Convolution parameters
        
    Returns:
        Output tensor of shape (B, C, H, W_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C, H, W = x.shape
    kernel_size = weight.shape[2]  # weight shape is (C, 1, kernel_size)
    
    # Calculate output width
    W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, H, W_out), dtype=x.dtype, device=x.device)
    
    # Set up kernel parameters
    kernel_params = {
        'BLOCK_SIZE_W': 64,
        'BLOCK_SIZE_C': 1,
    }
    
    # Grid: (batch_size, num_channels, height)
    grid = (B, C, H)
    
    # Determine if bias is present
    has_bias = bias is not None
    
    # Launch kernel
    depthwise_conv1d_kernel[grid](
        x, weight, bias if has_bias else None, out,
        B=B, C=C, H=H, W=W, W_out=W_out,
        kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        HAS_BIAS=has_bias,
        **kernel_params
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel.
    Optimized with custom Triton kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the same Conv2d layer structure but we'll override the forward pass
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create the weight and bias parameters (same as original)
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Call our Triton-based depthwise 1D convolution
        return triton_depthwise_conv1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )