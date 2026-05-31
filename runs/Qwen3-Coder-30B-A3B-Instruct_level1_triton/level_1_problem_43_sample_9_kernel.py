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
    input_d1,
    input_d2,
    input_d3,
    output_d1,
    output_d2,
    output_d3,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr
):
    # Get thread index
    idx = tl.program_id(0)
    
    # Calculate which output element this thread handles
    batch_idx = idx // (channels * output_d1 * output_d2 * output_d3)
    remaining = idx % (channels * output_d1 * output_d2 * output_d3)
    channel_idx = remaining // (output_d1 * output_d2 * output_d3)
    remaining = remaining % (output_d1 * output_d2 * output_d3)
    d1_idx = remaining // (output_d2 * output_d3)
    remaining = remaining % (output_d2 * output_d3)
    d2_idx = remaining // output_d3
    d3_idx = remaining % output_d3
    
    # Check bounds
    if batch_idx >= batch_size or channel_idx >= channels:
        return
        
    # Calculate input region boundaries
    start_d1 = d1_idx * stride - padding
    start_d2 = d2_idx * stride - padding
    start_d3 = d3_idx * stride - padding
    
    # Initialize max value
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    
    # Iterate through kernel
    for kd1 in range(kernel_size):
        for kd2 in range(kernel_size):
            for kd3 in range(kernel_size):
                # Apply dilation
                actual_d1 = start_d1 + kd1 * dilation
                actual_d2 = start_d2 + kd2 * dilation
                actual_d3 = start_d3 + kd3 * dilation
                
                # Check if within bounds
                if (actual_d1 >= 0 and actual_d1 < input_d1 and 
                    actual_d2 >= 0 and actual_d2 < input_d2 and 
                    actual_d3 >= 0 and actual_d3 < input_d3):
                    
                    # Calculate input index
                    input_idx = (batch_idx * (channels * input_d1 * input_d2 * input_d3) +
                                channel_idx * (input_d1 * input_d2 * input_d3) +
                                actual_d1 * (input_d2 * input_d3) +
                                actual_d2 * input_d3 +
                                actual_d3)
                    
                    val = tl.load(input_ptr + input_idx, mask=True)
                    max_val = tl.maximum(max_val, val)
    
    # Store result
    output_idx = (batch_idx * (channels * output_d1 * output_d2 * output_d3) +
                  channel_idx * (output_d1 * output_d2 * output_d3) +
                  d1_idx * (output_d2 * output_d3) +
                  d2_idx * output_d3 +
                  d3_idx)
    
    tl.store(output_ptr + output_idx, max_val)

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
        batch_size, channels, input_d1, input_d2, input_d3 = x.shape
        
        # Calculate output dimensions
        if self.ceil_mode:
            output_d1 = math.ceil((input_d1 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
            output_d2 = math.ceil((input_d2 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
            output_d3 = math.ceil((input_d3 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
        else:
            output_d1 = (input_d1 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
            output_d2 = (input_d2 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
            output_d3 = (input_d3 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
            
        # Ensure non-negative output sizes
        output_d1 = max(1, output_d1)
        output_d2 = max(1, output_d2)
        output_d3 = max(1, output_d3)
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_d1, output_d2, output_d3, device=x.device, dtype=torch.float32)
        
        # Handle case where output size is 1x1x1
        if output_d1 == 1 and output_d2 == 1 and output_d3 == 1:
            # For single element output, just find max across all input elements per batch/channel
            for b in range(batch_size):
                for c in range(channels):
                    # Flatten the input and find max
                    flat_input = x[b, c].flatten()
                    max_val = flat_input.max()
                    output[b, c, 0, 0, 0] = max_val
            return output
            
        # For larger outputs, use Triton kernel
        if batch_size * channels * output_d1 * output_d2 * output_d3 > 0:
            # Prepare input tensor for contiguous access
            x_contiguous = x.contiguous()
            
            # Grid configuration
            total_elements = batch_size * channels * output_d1 * output_d2 * output_d3
            BLOCK_SIZE = 1024
            
            # Determine grid size
            grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            # Launch kernel
            maxpool3d_kernel[grid](
                x_contiguous,
                output,
                batch_size,
                channels,
                input_d1,
                input_d2,
                input_d3,
                output_d1,
                output_d2,
                output_d3,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                BLOCK_SIZE
            )
            
        return output

# Helper function for backward compatibility
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