import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,              # Input tensor (B, C, H, W)
    w_ptr,              # Weight tensor (C, 1, K, K)
    b_ptr,              # Bias tensor (C,) - optional
    out_ptr,            # Output tensor (B, C, H_out, W_out)
    batch_size,         # B
    channels,           # C
    height_in,          # H
    width_in,           # W
    height_out,         # H_out
    width_out,          # W_out
    kernel_size,        # K
    stride,             # S
    padding,            # P
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs: batch, channel, and spatial positions
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    block_h_start = tl.program_id(2) * BLOCK_SIZE_H
    block_w_start = tl.program_id(3) * BLOCK_SIZE_W
    
    # Compute the actual spatial coordinates
    h_offsets = block_h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = block_w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Create meshgrid of output coordinates
    h_ids, w_ids = tl.meshgrid(h_offsets, w_offsets)
    h_ids = h_ids.T
    w_ids = w_ids.T
    
    # Compute input coordinates (with padding offset)
    h_in = h_ids * stride - padding
    w_in = w_ids * stride - padding
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Compute input positions
            h_input = h_in + kh
            w_input = w_in + kw
            
            # Check bounds for valid input positions
            valid = (h_input >= 0) & (h_input < height_in) & (w_input >= 0) & (w_input < width_in)
            
            # Calculate input pointer offset for valid positions
            # Memory layout: (batch, channel, height, width)
            input_offset = (batch_id * channels * height_in * width_in + 
                          channel_id * height_in * width_in + 
                          h_input * width_in + w_input)
            
            # Load input values (0 if out of bounds)
            x_val = tl.load(x_ptr + input_offset, mask=valid, other=0.0)
            
            # Calculate weight pointer offset
            # Weight layout: (out_channels, in_channels_per_group, kernel_h, kernel_w)
            # For depthwise: out_channels = in_channels, in_channels_per_group = 1
            weight_offset = (channel_id * kernel_size * kernel_size + 
                           kh * kernel_size + kw)
            w_val = tl.load(w_ptr + weight_offset)
            
            # Accumulate
            acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + channel_id)
        acc += bias
    
    # Store result
    output_offset = (batch_id * channels * height_out * width_out + 
                   channel_id * height_out * width_out + 
                   h_ids * width_out + w_ids)
    tl.store(out_ptr + output_offset, acc, mask=(h_ids < height_out) & (w_ids < width_out))


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height_in, width_in)
        weight: Weight tensor of shape (in_channels, 1, kernel_size, kernel_size)
        bias: Optional bias tensor of shape (in_channels,)
        stride: Stride of convolution
        padding: Padding applied to input
    """
    batch_size, channels, height_in, width_in = x.shape
    _, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in + 2 * padding - kernel_size) // stride + 1
    width_out = (width_in + 2 * padding - kernel_size) // stride + 1
    
    # Allocate output tensor
    out = torch.empty((batch_size, channels, height_out, width_out), 
                     dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_K = 3  # For typical kernel sizes
    
    grid = (
        batch_size,  # batch dimension
        channels,    # channel dimension
        (height_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,  # h blocks
        (width_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W    # w blocks
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, channels,
        height_in, width_in,
        height_out, width_out,
        kernel_size, stride, padding,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using custom Triton kernel.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # For depthwise convolution, groups = in_channels, out_channels must equal in_channels
        # Store parameters for reference but use our custom kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create weight and bias parameters with proper shapes for depthwise conv
        # Weight shape: (out_channels, in_channels // groups, kernel_size, kernel_size)
        # For depthwise: groups = in_channels, so in_channels // groups = 1
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights (similar to PyTorch default initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        import math
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Call our custom Triton kernel
        return triton_depthwise_conv2d(
            x, self.weight, self.bias, 
            stride=self.stride, padding=self.padding
        )