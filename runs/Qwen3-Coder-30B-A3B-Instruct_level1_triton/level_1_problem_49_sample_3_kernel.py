import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_reduce_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_input_inner,
    stride_input_outer,
    stride_output,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID for the outer dimension
    outer_idx = tl.program_id(0)
    
    # Calculate the starting position for this program
    input_start = outer_idx * stride_input_outer
    
    # Shared memory for reduction
    shared_max = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Loop over the inner dimension
    for i in range(0, dim_size, BLOCK_SIZE):
        # Calculate current offset
        current_offset = i + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < dim_size
        
        # Load data from global memory
        input_offset = input_start + current_offset * stride_input_inner
        x = tl.load(input_ptr + input_offset, mask=mask, other=-float('inf'))
        
        # Store in shared memory
        tl.store(shared_max + current_offset, x, mask=mask)
        
        # Synchronize threads to ensure all data is loaded
        tl.sync()
        
        # Perform reduction in shared memory
        max_val = tl.reduce(shared_max, axis=0, combine_fn=tl.maximum)
        
        # Store result
        if i == 0:
            tl.store(output_ptr + outer_idx * stride_output, max_val)
        else:
            # For subsequent blocks, we need to compare with existing value
            old_max = tl.load(output_ptr + outer_idx * stride_output)
            new_max = tl.maximum(old_max, max_val)
            tl.store(output_ptr + outer_idx * stride_output, new_max)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Calculate output shape
        output_shape = list(x.shape)
        del output_shape[self.dim]
        
        # Create output tensor
        output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
        
        # Handle different dimensions
        if self.dim == 0:
            # Reduce along dimension 0
            n_elements = x.numel() // x.shape[0]
            dim_size = x.shape[0]
            stride_input_inner = x.stride(0)
            stride_input_outer = x.stride(1) if len(x.shape) > 1 else 1
            stride_output = 1
            
            # Launch kernel
            grid = (x.shape[1] if len(x.shape) > 1 else 1,)
            max_reduce_kernel[grid](
                x,
                output,
                n_elements,
                dim_size,
                stride_input_inner,
                stride_input_outer,
                stride_output,
                BLOCK_SIZE=1024
            )
            
        elif self.dim == 1:
            # Reduce along dimension 1
            n_elements = x.numel() // x.shape[1]
            dim_size = x.shape[1]
            stride_input_inner = x.stride(1)
            stride_input_outer = x.stride(0)
            stride_output = 1
            
            # Launch kernel
            grid = (x.shape[0],)
            max_reduce_kernel[grid](
                x,
                output,
                n_elements,
                dim_size,
                stride_input_inner,
                stride_input_outer,
                stride_output,
                BLOCK_SIZE=1024
            )
            
        else:
            # For other dimensions, use PyTorch implementation
            return torch.max(x, dim=self.dim)[0]
            
        return output