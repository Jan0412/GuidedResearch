import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_input_batch,
    stride_input_dim,
    stride_output_batch,
    stride_output_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting position for this batch
    input_base = batch_idx * stride_input_batch
    output_base = batch_idx * stride_output_batch
    
    # Process each element along the reduced dimension
    for i in range(0, dim_size, BLOCK_SIZE):
        # Create offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        
        # Create mask to avoid going out of bounds
        mask = offsets < dim_size
        
        # Load input data
        input_vals = tl.load(input_ptr + input_base + offsets * stride_input_dim, mask=mask, other=-float('inf'))
        
        # Find maximum value in this block
        max_val = tl.max(input_vals)
        
        # Store the maximum value
        tl.store(output_ptr + output_base + (offsets // dim_size) * stride_output_dim, max_val, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        shape = x.shape
        batch_size = shape[0]
        dim_size = shape[self.dim]
        
        # Compute output shape
        output_shape = list(shape)
        output_shape.pop(self.dim)
        
        # Create output tensor
        output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
        
        # Handle different dimensions
        if self.dim == 0:
            # Reduce along dimension 0
            stride_input_batch = x.stride(0)
            stride_input_dim = x.stride(1) if len(x.shape) > 1 else 1
            stride_output_batch = 0
            stride_output_dim = 1
            
            # Launch kernel
            grid = (batch_size,)
            max_reduction_kernel[grid](
                x, 
                output, 
                x.numel(), 
                dim_size, 
                stride_input_batch, 
                stride_input_dim, 
                stride_output_batch, 
                stride_output_dim, 
                BLOCK_SIZE=128
            )
            
        elif self.dim == 1:
            # Reduce along dimension 1
            stride_input_batch = x.stride(0)
            stride_input_dim = x.stride(1)
            stride_output_batch = 1
            stride_output_dim = 0
            
            # Launch kernel
            grid = (batch_size,)
            max_reduction_kernel[grid](
                x, 
                output, 
                x.numel(), 
                dim_size, 
                stride_input_batch, 
                stride_input_dim, 
                stride_output_batch, 
                stride_output_dim, 
                BLOCK_SIZE=128
            )
        else:
            # For other dimensions, use a more general approach
            # This implementation assumes we're reducing the last dimension for simplicity
            # In practice, you'd want a more sophisticated approach to handle all cases
            return torch.max(x, dim=self.dim)[0]
            
        return output