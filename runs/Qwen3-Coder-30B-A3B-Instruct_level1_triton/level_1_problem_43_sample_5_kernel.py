import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def maxpool3d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_d1, input_d2, input_d3,
    output_d1, output_d2, output_d3,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread index
    idx = tl.program_id(0)
    
    # Calculate which output element this thread handles
    # Flatten output indices
    output_idx = idx
    
    # Convert flat index to 3D coordinates (b, c, d1, d2, d3)
    d3_out = output_idx % output_d3
    remaining = output_idx // output_d3
    d2_out = remaining % output_d2
    remaining = remaining // output_d2
    d1_out = remaining % output_d1
    remaining = remaining // output_d1
    c = remaining % channels
    b = remaining // channels
    
    # Calculate input region boundaries
    d1_in_start = d1_out * stride - padding
    d2_in_start = d2_out * stride - padding
    d3_in_start = d3_out * stride - padding
    
    # Apply dilation
    d1_in_start = d1_in_start * dilation
    d2_in_start = d2_in_start * dilation
    d3_in_start = d3_in_start * dilation
    
    # Initialize max value
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    
    # Iterate over kernel
    for kd1 in range(kernel_size):
        for kd2 in range(kernel_size):
            for kd3 in range(kernel_size):
                # Calculate input position
                d1_in = d1_in_start + kd1 * dilation
                d2_in = d2_in_start + kd2 * dilation
                d3_in = d3_in_start + kd3 * dilation
                
                # Check bounds
                if (d1_in >= 0 and d1_in < input_d1 and 
                    d2_in >= 0 and d2_in < input_d2 and 
                    d3_in >= 0 and d3_in < input_d3):
                    
                    # Calculate input offset
                    input_offset = (b * channels + c) * (input_d1 * input_d2 * input_d3) + \
                                   (d1_in * input_d2 * input_d3 + d2_in * input_d3 + d3_in)
                    
                    # Load value and update max
                    val = tl.load(input_ptr + input_offset, mask=True)
                    max_val = tl.maximum(max_val, val)
    
    # Store result
    output_offset = (b * channels + c) * (output_d1 * output_d2 * output_d3) + \
                    (d1_out * output_d2 * output_d3 + d2_out * output_d3 + d3_out)
    tl.store(output_ptr + output_offset, max_val)

class ModelNew(nn.Module):
    """
    Optimized version of Max Pooling 3D using Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
    def _calculate_output_size(self, input_size, kernel_size, stride, padding, dilation):
        """Calculate output size for max pooling."""
        if self.ceil_mode:
            return math.ceil((input_size + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
        else:
            return (input_size + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        batch_size, channels, dim1, dim2, dim3 = x.shape
        
        # Calculate output dimensions
        output_d1 = self._calculate_output_size(dim1, self.kernel_size, self.stride, self.padding, self.dilation)
        output_d2 = self._calculate_output_size(dim2, self.kernel_size, self.stride, self.padding, self.dilation)
        output_d3 = self._calculate_output_size(dim3, self.kernel_size, self.stride, self.padding, self.dilation)
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_d1, output_d2, output_d3, device=x.device, dtype=torch.float32)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Calculate total number of output elements
        total_elements = batch_size * channels * output_d1 * output_d2 * output_d3
        
        if total_elements > 0:
            # Configure grid size
            BLOCK_SIZE = 128
            grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            # Launch kernel
            maxpool3d_kernel[grid_size](
                x,
                output,
                batch_size,
                channels,
                dim1, dim2, dim3,
                output_d1, output_d2, output_d3,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                BLOCK_SIZE=BLOCK_SIZE
            )
        
        return output

# For backward compatibility with original interface
def get_inputs():
    batch_size = 16
    channels = 32
    dim1 = 128
    dim2 = 128
    dim3 = 128
    x = torch.rand(batch_size, channels, dim1, dim2, dim3)
    return [x]

def get_init_inputs():
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 3
    return [kernel_size, stride, padding, dilation]