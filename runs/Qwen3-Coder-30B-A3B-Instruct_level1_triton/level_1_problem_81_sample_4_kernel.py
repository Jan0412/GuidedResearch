import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)
    
    # Calculate output position
    if out_y >= output_height or out_x >= output_width:
        return
    
    # Shared memory for input tile
    tile_size = BLOCK_SIZE
    input_tile = tl.shared_tile(input_ptr, [tile_size, tile_size], [1, 1])
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel elements and input channels
    for k in range(in_channels):
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                # Compute input coordinates
                input_y = out_y * stride - padding + ky * dilation
                input_x = out_x * stride - padding + kx * dilation
                
                # Check bounds
                if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_idx * in_channels * input_height * input_width +
                                       k * input_height * input_width +
                                       input_y * input_width +
                                       input_x)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr +
                                        out_channel_idx * in_channels * kernel_size * kernel_size +
                                        k * kernel_size * kernel_size +
                                        ky * kernel_size +
                                        kx)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + 
             batch_idx * out_channels * output_height * output_width +
             out_channel_idx * output_height * output_width +
             out_y * output_width +
             out_x,
             acc[0])

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Triton implementation of 2D transposed convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    output_width = (input_width - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block size
    BLOCK_SIZE = 16
    
    # Grid dimensions
    grid = (
        batch_size,           # Batch dimension
        out_channels,         # Output channel dimension
        (output_height + BLOCK_SIZE - 1) // BLOCK_SIZE,  # Output height tiles
        (output_width + BLOCK_SIZE - 1) // BLOCK_SIZE   # Output width tiles
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and square kernel, supporting dilation, padding, and stride.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in). 

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use the Triton implementation
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )

# For backward compatibility, keep the original class name as well
Model = ModelNew