import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride_input_inner,
    stride_input_outer,
    stride_output_inner,
    stride_output_outer,
    BLOCK_SIZE: tl.constexpr,
    DIM_SIZE: tl.constexpr
):
    # Get the block ID
    block_id = tl.program_id(0)
    
    # Calculate the starting position for this block
    start_idx = block_id * BLOCK_SIZE
    
    # Calculate the total number of elements in the output
    total_output_elements = n_elements // DIM_SIZE
    
    # Iterate through the output dimensions
    for i in range(total_output_elements):
        # Calculate the offset for this output element
        output_offset = i
        
        # Initialize maximum value
        max_val = tl.full([], float('-inf'), dtype=tl.float32)
        
        # Process elements along the reduction dimension
        for j in range(DIM_SIZE):
            # Calculate input index
            input_idx = i * stride_input_inner + j * stride_input_outer
            
            # Load input value
            val = tl.load(input_ptr + input_idx, mask=(input_idx < n_elements))
            
            # Update maximum
            max_val = tl.maximum(max_val, val)
        
        # Store result
        tl.store(output_ptr + output_offset, max_val)

@triton.jit
def fused_max_reduce_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim1,
    dim2,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the grid size
    grid_size = batch_size * dim1
    
    # Each thread block processes one output element
    block_id = tl.program_id(0)
    
    if block_id >= grid_size:
        return
    
    # Calculate which batch and dim1 we're working on
    batch_idx = block_id // dim1
    dim1_idx = block_id % dim1
    
    # Initialize max value
    max_val = tl.full([], float('-inf'), dtype=tl.float32)
    
    # Iterate through all elements in dim2 for this batch and dim1
    for i in range(dim2):
        # Calculate input index: [batch][dim1][i]
        input_idx = batch_idx * (dim1 * dim2) + dim1_idx * dim2 + i
        
        # Load value
        val = tl.load(input_ptr + input_idx)
        
        # Update maximum
        max_val = tl.maximum(max_val, val)
    
    # Store result
    output_idx = batch_idx * dim1 + dim1_idx
    tl.store(output_ptr + output_idx, max_val)

class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using Triton kernels.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        # For simplicity, let's assume we're reducing over the last dimension (dim2)
        # This matches the example dimensions provided
        
        if self.dim != 2:
            # Fall back to standard PyTorch implementation for non-optimized cases
            return torch.max(x, dim=self.dim)[0]
        
        # Optimize for the case where we reduce over the last dimension
        batch_size, dim1, dim2 = x.shape
        
        # Prepare output tensor
        output_shape = list(x.shape)
        output_shape.pop(self.dim)
        output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
        
        # Ensure input is contiguous and on GPU
        x = x.contiguous()
        
        # Calculate grid size
        grid_size = batch_size * dim1
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Launch kernel
        fused_max_reduce_kernel[grid_size](x, output, batch_size, dim1, dim2, BLOCK_SIZE)
        
        return output