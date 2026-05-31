import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    reduce_dim_size,
    stride_input_reduce,
    stride_output_reduce,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets * stride_input_reduce, mask=mask, other=0.0)
    
    # Reduce along the specified dimension
    # For each element, we need to sum across the reduce_dim_size dimension
    # We'll use a simple approach where we accumulate in registers
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over reduce dimension
    for i in range(reduce_dim_size):
        # Calculate offset for current element in reduce dimension
        current_offset = offsets * stride_input_reduce + i * stride_input_reduce
        # Load value
        val = tl.load(input_ptr + current_offset, mask=mask, other=0.0)
        # Accumulate
        accumulator += val
    
    # Store result
    tl.store(output_ptr + offsets * stride_output_reduce, accumulator, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for sum reduction.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # Ensure tensor is contiguous for efficient memory access
        x = x.contiguous()
        
        # Get dimensions
        dims = list(x.shape)
        reduce_dim_size = dims[self.dim]
        
        # Calculate output shape
        output_shape = dims.copy()
        output_shape[self.dim] = 1
        
        # Create output tensor
        output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
        
        # Calculate strides
        stride_input_reduce = 1
        stride_output_reduce = 1
        
        # Compute strides for the reduce dimension
        for i in range(self.dim):
            stride_input_reduce *= dims[i]
            
        for i in range(self.dim + 1, len(dims)):
            stride_input_reduce *= dims[i]
            
        # Compute stride for output
        for i in range(self.dim):
            stride_output_reduce *= dims[i]
            
        # Handle special case when reduce_dim is last dimension
        if self.dim == len(dims) - 1:
            stride_input_reduce = 1
            stride_output_reduce = 1
            for i in range(len(dims) - 1):
                stride_output_reduce *= dims[i]
        else:
            # Compute stride for the reduced dimension
            stride_input_reduce = 1
            for i in range(self.dim + 1, len(dims)):
                stride_input_reduce *= dims[i]
                
            # Compute stride for output
            stride_output_reduce = 1
            for i in range(self.dim):
                stride_output_reduce *= dims[i]
                
        # Flatten input and output for easier processing
        flat_input = x.view(-1, reduce_dim_size)
        flat_output = output.view(-1, 1)
        
        # Get total number of elements in flattened input
        n_elements = flat_input.numel() // reduce_dim_size
        
        if n_elements == 0:
            return output
            
        BLOCK_SIZE = 1024
        
        # Grid size calculation
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        
        # Launch kernel
        sum_reduction_kernel[grid](
            flat_input,
            flat_output,
            n_elements,
            reduce_dim_size,
            stride_input_reduce,
            stride_output_reduce,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output