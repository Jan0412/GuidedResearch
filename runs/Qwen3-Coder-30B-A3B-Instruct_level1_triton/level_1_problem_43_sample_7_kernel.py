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
    output_idx = idx
    
    # Convert linear index to 3D coordinates
    out_d3 = output_idx % output_d3
    remaining = output_idx // output_d3
    out_d2 = remaining % output_d2
    remaining = remaining // output_d2
    out_d1 = remaining % output_d1
    remaining = remaining // output_d1
    channel = remaining % channels
    batch = remaining // channels
    
    # Calculate input region boundaries
    d1_start = out_d1 * stride - padding
    d2_start = out_d2 * stride - padding
    d3_start = out_d3 * stride - padding
    
    # Initialize maximum value
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    
    # Iterate through kernel
    for kd in range(kernel_size):
        for k2 in range(kernel_size):
            for k3 in range(kernel_size):
                # Apply dilation
                d1 = d1_start + kd * dilation
                d2 = d2_start + k2 * dilation
                d3 = d3_start + k3 * dilation
                
                # Check bounds
                if (d1 >= 0 and d1 < input_d1 and 
                    d2 >= 0 and d2 < input_d2 and 
                    d3 >= 0 and d3 < input_d3):
                    
                    # Calculate input index
                    input_idx = batch * (channels * input_d1 * input_d2 * input_d3) + \
                               channel * (input_d1 * input_d2 * input_d3) + \
                               d1 * (input_d2 * input_d3) + \
                               d2 * input_d3 + \
                               d3
                    
                    val = tl.load(input_ptr + input_idx, mask=True)
                    max_val = tl.maximum(max_val, val)
    
    # Store result
    output_idx = batch * (channels * output_d1 * output_d2 * output_d3) + \
                 channel * (output_d1 * output_d2 * output_d3) + \
                 out_d1 * (output_d2 * output_d3) + \
                 out_d2 * output_d3 + \
                 out_d3
    
    tl.store(output_ptr + output_idx, max_val)

def calculate_output_size(input_size, kernel_size, stride, padding, dilation):
    """Calculate output size for pooling operation"""
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    return math.floor((input_size + 2 * padding - effective_kernel_size) / stride + 1)

class ModelNew(nn.Module):
    """
    Optimized Max Pooling 3D using Triton kernels.
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
        output_d1 = calculate_output_size(dim1, self.kernel_size, self.stride, self.padding, self.dilation)
        output_d2 = calculate_output_size(dim2, self.kernel_size, self.stride, self.padding, self.dilation)
        output_d3 = calculate_output_size(dim3, self.kernel_size, self.stride, self.padding, self.dilation)
        
        # Prepare output tensor
        output = torch.empty(batch_size, channels, output_d1, output_d2, output_d3, device=x.device, dtype=torch.float32)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Calculate total elements in output
        total_elements = batch_size * channels * output_d1 * output_d2 * output_d3
        
        if total_elements == 0:
            return output
            
        # Set up kernel launch parameters
        BLOCK_SIZE = 1024
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

# Dummy functions to match the interface
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