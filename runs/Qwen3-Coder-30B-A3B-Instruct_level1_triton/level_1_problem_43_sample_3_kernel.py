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
    kernel_d1,
    kernel_d2,
    kernel_d3,
    stride_d1,
    stride_d2,
    stride_d3,
    padding_d1,
    padding_d2,
    padding_d3,
    dilation_d1,
    dilation_d2,
    dilation_d3,
    return_indices,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread index
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
    
    # Calculate input start positions
    input_start_d1 = d1_idx * stride_d1 - padding_d1
    input_start_d2 = d2_idx * stride_d2 - padding_d2
    input_start_d3 = d3_idx * stride_d3 - padding_d3
    
    # Initialize max value and index
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Iterate through kernel
    for kd1 in range(kernel_d1):
        for kd2 in range(kernel_d2):
            for kd3 in range(kernel_d3):
                # Calculate actual input position
                input_d1_pos = input_start_d1 + kd1 * dilation_d1
                input_d2_pos = input_start_d2 + kd2 * dilation_d2
                input_d3_pos = input_start_d3 + kd3 * dilation_d3
                
                # Check bounds
                if (input_d1_pos >= 0 and input_d1_pos < input_d1 and
                    input_d2_pos >= 0 and input_d2_pos < input_d2 and
                    input_d3_pos >= 0 and input_d3_pos < input_d3):
                    
                    # Calculate linear index in input tensor
                    input_idx = (
                        batch_idx * (channels * input_d1 * input_d2 * input_d3) +
                        channel_idx * (input_d1 * input_d2 * input_d3) +
                        input_d1_pos * (input_d2 * input_d3) +
                        input_d2_pos * input_d3 +
                        input_d3_pos
                    )
                    
                    # Load value from input
                    val = tl.load(input_ptr + input_idx, mask=True)
                    
                    # Update max
                    mask = val > max_val
                    max_val = tl.where(mask, val, max_val)
                    if return_indices:
                        max_idx = tl.where(mask, input_idx, max_idx)
    
    # Write output
    output_idx = (
        batch_idx * (channels * output_d1 * output_d2 * output_d3) +
        channel_idx * (output_d1 * output_d2 * output_d3) +
        d1_idx * (output_d2 * output_d3) +
        d2_idx * output_d3 +
        d3_idx
    )
    
    tl.store(output_ptr + output_idx, max_val)
    
    if return_indices:
        tl.store(indices_ptr + output_idx, max_idx)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, input_d1, input_d2, input_d3 = x.shape
        
        # Compute output dimensions
        if self.ceil_mode:
            output_d1 = math.ceil((input_d1 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
            output_d2 = math.ceil((input_d2 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
            output_d3 = math.ceil((input_d3 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
        else:
            output_d1 = (input_d1 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
            output_d2 = (input_d2 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
            output_d3 = (input_d3 + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
            
        # Ensure non-negative output dimensions
        output_d1 = max(1, output_d1)
        output_d2 = max(1, output_d2)
        output_d3 = max(1, output_d3)
        
        # Create output tensors
        output = torch.empty(batch_size, channels, output_d1, output_d2, output_d3, device=x.device, dtype=torch.float32)
        
        if self.return_indices:
            indices = torch.empty(batch_size, channels, output_d1, output_d2, output_d3, device=x.device, dtype=torch.int32)
        
        # Prepare kernel parameters
        kernel_d1 = kernel_d2 = kernel_d3 = self.kernel_size
        stride_d1 = stride_d2 = stride_d3 = self.stride
        padding_d1 = padding_d2 = padding_d3 = self.padding
        dilation_d1 = dilation_d2 = dilation_d3 = self.dilation
        
        # Handle case where input is smaller than kernel
        if output_d1 == 1 and output_d2 == 1 and output_d3 == 1:
            # For very small outputs, use a simpler approach
            return self._simple_maxpool3d(x)
        
        # Flatten output for indexing
        total_elements = batch_size * channels * output_d1 * output_d2 * output_d3
        
        if total_elements > 0:
            # Set up grid
            BLOCK_SIZE = 1024
            grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            # Launch kernel
            if self.return_indices:
                maxpool3d_kernel[grid](
                    x, output, indices,
                    batch_size, channels, input_d1, input_d2, input_d3,
                    output_d1, output_d2, output_d3,
                    kernel_d1, kernel_d2, kernel_d3,
                    stride_d1, stride_d2, stride_d3,
                    padding_d1, padding_d2, padding_d3,
                    dilation_d1, dilation_d2, dilation_d3,
                    self.return_indices,
                    BLOCK_SIZE=BLOCK_SIZE
                )
                return output, indices
            else:
                maxpool3d_kernel[grid](
                    x, output, None,
                    batch_size, channels, input_d1, input_d2, input_d3,
                    output_d1, output_d2, output_d3,
                    kernel_d1, kernel_d2, kernel_d3,
                    stride_d1, stride_d2, stride_d3,
                    padding_d1, padding_d2, padding_d3,
                    dilation_d1, dilation_d2, dilation_d3,
                    self.return_indices,
                    BLOCK_SIZE=BLOCK_SIZE
                )
                return output
        else:
            return output
    
    def _simple_maxpool3d(self, x):
        # Fallback for cases with small output sizes
        batch_size, channels, input_d1, input_d2, input_d3 = x.shape
        output = torch.empty_like(x[:, :, :1, :1, :1])
        
        # Simple loop implementation for edge cases
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output.shape[2]):
                    for j in range(output.shape[3]):
                        for k in range(output.shape[4]):
                            # Compute pooling window bounds
                            start_i = max(0, i * self.stride - self.padding)
                            end_i = min(input_d1, start_i + self.kernel_size * self.dilation)
                            start_j = max(0, j * self.stride - self.padding)
                            end_j = min(input_d2, start_j + self.kernel_size * self.dilation)
                            start_k = max(0, k * self.stride - self.padding)
                            end_k = min(input_d3, start_k + self.kernel_size * self.dilation)
                            
                            # Extract window and find max
                            window = x[b, c, start_i:end_i:self.dilation, 
                                     start_j:end_j:self.dilation, 
                                     start_k:end_k:self.dilation]
                            output[b, c, i, j, k] = window.max()
        
        return output