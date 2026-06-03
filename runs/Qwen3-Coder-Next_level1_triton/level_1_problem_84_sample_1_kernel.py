import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor: (B, C, H, W)
    w_ptr,  # Weight tensor: (C, 1, kH, kW)
    b_ptr,  # Bias tensor: (C,) or None
    out_ptr,  # Output tensor: (B, C, H_out, W_out)
    batch_size, n_channels, in_height, in_width,
    out_height, out_width,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Batch and channel indices
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    
    # Calculate output spatial indices
    w_out_idx = tl.program_id(2)
    h_out_idx = w_out_idx // out_width
    w_out_idx = w_out_idx % out_width
    
    # Calculate input starting position for this output position
    h_in_start = h_out_idx * stride - padding
    w_in_start = w_out_idx * stride - padding
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over kernel
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            h_in = h_in_start + kh
            w_in = w_in_start + kw
            
            # Check if within bounds
            in_bounds = (h_in >= 0) & (h_in < in_height) & (w_in >= 0) & (w_in < in_width)
            
            if in_bounds:
                # Calculate input pointer offset
                input_offset = (b_idx * n_channels * in_height * in_width + 
                              c_idx * in_height * in_width + 
                              h_in * in_width + w_in)
                
                # Calculate weight pointer offset
                weight_offset = c_idx * kernel_size * kernel_size + kh * kernel_size + kw
                
                # Load and multiply
                x_val = tl.load(x_ptr + input_offset)
                w_val = tl.load(w_ptr + weight_offset)
                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_offset = c_idx
        acc += tl.load(b_ptr + bias_offset)
    
    # Calculate output pointer offset
    output_offset = (b_idx * n_channels * out_height * out_width + 
                   c_idx * out_height * out_width + 
                   h_out_idx * out_width + w_out_idx)
    
    # Store result
    tl.store(out_ptr + output_offset, acc)


def triton_depthwise_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0):
    """
    Performs depthwise 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Weight tensor of shape (C, 1, kH, kW)
        bias: Optional bias tensor of shape (C,)
        stride: Stride of convolution
        padding: Padding applied to input
        
    Returns:
        Output tensor of shape (B, C, H_out, W_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    batch_size, n_channels, in_height, in_width = x.shape
    _, _, kernel_size_h, kernel_size_w = weight.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - kernel_size_h) // stride + 1
    out_height = max(0, out_height)
    out_width = (in_width + 2 * padding - kernel_size_w) // stride + 1
    out_width = max(0, out_width)
    
    # Prepare output tensor
    out = torch.empty(batch_size, n_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Grid configuration
    # Grid: (batch_size, channels, height_out * width_out)
    grid = (batch_size, n_channels, out_height * out_width)
    
    # Launch kernel
    BLOCK_SIZE = 1  # Not used in current implementation but kept for API compatibility
    
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, n_channels, in_height, in_width,
        out_height, out_width,
        kernel_size_h, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create weight and bias parameters
        # For depthwise convolution, weight shape is (out_channels, 1, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)