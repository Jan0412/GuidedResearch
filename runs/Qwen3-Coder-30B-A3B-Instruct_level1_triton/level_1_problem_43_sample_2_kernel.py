import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def maxpool3d_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
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
    return_indices,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread index
    idx = tl.program_id(0)
    
    # Calculate which batch and channel this thread handles
    total_elements = batch_size * channels * output_d1 * output_d2 * output_d3
    if idx >= total_elements:
        return
        
    # Decompose the linear index into multi-dimensional indices
    elem_idx = idx
    out_d3_idx = elem_idx % output_d3
    elem_idx //= output_d3
    out_d2_idx = elem_idx % output_d2
    elem_idx //= output_d2
    out_d1_idx = elem_idx % output_d1
    elem_idx //= output_d1
    channel_idx = elem_idx % channels
    batch_idx = elem_idx // channels
    
    # Calculate the starting position in the input tensor for this pooling window
    start_d1 = out_d1_idx * stride - padding
    start_d2 = out_d2_idx * stride - padding
    start_d3 = out_d3_idx * stride - padding
    
    # Initialize max value and index
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Iterate through the kernel
    for kd in range(kernel_size):
        for k2 in range(kernel_size):
            for k3 in range(kernel_size):
                # Calculate input coordinates with dilation
                input_d1_pos = start_d1 + kd * dilation
                input_d2_pos = start_d2 + k2 * dilation
                input_d3_pos = start_d3 + k3 * dilation
                
                # Check bounds
                if (input_d1_pos >= 0 and input_d1_pos < input_d1 and 
                    input_d2_pos >= 0 and input_d2_pos < input_d2 and 
                    input_d3_pos >= 0 and input_d3_pos < input_d3):
                    
                    # Calculate input index
                    input_idx = (
                        batch_idx * (channels * input_d1 * input_d2 * input_d3) +
                        channel_idx * (input_d1 * input_d2 * input_d3) +
                        input_d1_pos * (input_d2 * input_d3) +
                        input_d2_pos * input_d3 +
                        input_d3_pos
                    )
                    
                    # Load input value
                    val = tl.load(input_ptr + input_idx, mask=True, other=float('-inf'))
                    
                    # Update max
                    if val > max_val:
                        max_val = val
                        if return_indices:
                            max_idx = tl.full([1], input_idx, dtype=tl.int32)
    
    # Write output
    output_idx = (
        batch_idx * (channels * output_d1 * output_d2 * output_d3) +
        channel_idx * (output_d1 * output_d2 * output_d3) +
        out_d1_idx * (output_d2 * output_d3) +
        out_d2_idx * output_d3 +
        out_d3_idx
    )
    
    tl.store(output_ptr + output_idx, max_val)
    
    if return_indices:
        tl.store(indices_ptr + output_idx, max_idx)

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
        if self.ceil_mode:
            output_d1 = math.ceil((dim1 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
            output_d2 = math.ceil((dim2 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
            output_d3 = math.ceil((dim3 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
        else:
            output_d1 = (dim1 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
            output_d2 = (dim2 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
            output_d3 = (dim3 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
            
        # Ensure non-negative output sizes
        output_d1 = max(1, output_d1)
        output_d2 = max(1, output_d2)
        output_d3 = max(1, output_d3)
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_d1, output_d2, output_d3, device=x.device, dtype=torch.float32)
        
        if self.return_indices:
            indices = torch.empty(batch_size, channels, output_d1, output_d2, output_d3, device=x.device, dtype=torch.int32)
            
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Calculate total elements for kernel launch
        total_elements = batch_size * channels * output_d1 * output_d2 * output_d3
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Grid size calculation
        grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        if self.return_indices:
            maxpool3d_kernel[grid_size](
                x,
                output,
                indices,
                batch_size,
                channels,
                dim1,
                dim2,
                dim3,
                output_d1,
                output_d2,
                output_d3,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.return_indices,
                BLOCK_SIZE=BLOCK_SIZE
            )
            return output, indices
        else:
            maxpool3d_kernel[grid_size](
                x,
                output,
                None,  # No indices pointer when not needed
                batch_size,
                channels,
                dim1,
                dim2,
                dim3,
                output_d1,
                output_d2,
                output_d3,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.return_indices,
                BLOCK_SIZE=BLOCK_SIZE
            )
            return output