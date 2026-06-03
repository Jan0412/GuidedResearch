import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias tensor pointer (optional)
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    input_height,  # Input height
    input_width,  # Input width
    kernel_size,  # Kernel size
    stride,  # Stride
    padding,  # Padding
    output_height,  # Output height
    output_width,  # Output width
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get batch, channel, and spatial indices
    bc = tl.program_id(0)
    bh = tl.program_id(1)
    bw = tl.program_id(2)
    
    batch_idx = bc // in_channels
    channel_idx = bc % in_channels
    
    # Compute output spatial position
    h_out = bh
    w_out = bw
    
    # Compute input starting position
    h_in = h_out * stride - padding
    w_in = w_out * stride - padding
    
    # Accumulator for the convolution result
    acc = 0.0
    
    # Iterate over kernel dimensions
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            h = h_in + kh
            w = w_in + kw
            
            # Check bounds
            if 0 <= h < input_height and 0 <= w < input_width:
                # Compute input index
                input_idx = batch_idx * in_channels * input_height * input_width + \
                           channel_idx * input_height * input_width + \
                           h * input_width + w
                # Load input value
                x_val = tl.load(x_ptr + input_idx)
                
                # Compute weight index
                weight_idx = channel_idx * kernel_size * kernel_size + \
                            kh * kernel_size + kw
                # Load weight value
                w_val = tl.load(w_ptr + weight_idx)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_idx = channel_idx
        acc += tl.load(b_ptr + bias_idx)
    
    # Compute output index
    out_idx = batch_idx * in_channels * output_height * output_width + \
             channel_idx * output_height * output_width + \
             h_out * output_width + w_out
    
    # Store result
    tl.store(out_ptr + out_idx, acc)


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0):
    """
    Performs depthwise 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (in_channels, kernel_size, kernel_size)
        bias: Optional bias tensor of shape (in_channels,)
        stride: Stride of convolution
        padding: Padding applied to input
        
    Returns:
        Output tensor of shape (batch_size, in_channels, output_height, output_width)
    """
    batch_size, in_channels, input_height, input_width = x.shape
    _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, in_channels, output_height, output_width, 
                     dtype=x.dtype, device=x.device)
    
    # Check if input and weight are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Grid configuration: (batch_size * in_channels, output_height, output_width)
    grid = (batch_size * in_channels, output_height, output_width)
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, input_height, input_width,
        kernel_size, stride, padding, output_height, output_width,
        BLOCK_SIZE_H=1,
        BLOCK_SIZE_W=1,
        BLOCK_K=1,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size, kernel_size))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_buffer('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 2D convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return triton_depthwise_conv2d(x, self.weight, self.bias, 
                                      self.stride, self.padding)