import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduce_kernel(
    input_ptr,
    output_ptr,
    reduce_dim_size,
    other_dims_size,
    stride_input_reduce,
    stride_output_reduce,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block ID for the reduction dimension
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Calculate the total number of elements in the output
    num_output_elements = other_dims_size
    
    # Loop over all output elements
    for i in range(tl.cdiv(num_output_elements, BLOCK_SIZE)):
        # Calculate global output index
        output_idx = i * BLOCK_SIZE + tl.program_id(1)
        
        if output_idx < num_output_elements:
            # Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            
            # Reduce along the specified dimension
            for j in range(tl.cdiv(reduce_dim_size, BLOCK_SIZE)):
                # Calculate input indices
                input_offset = output_idx * stride_input_reduce + j * BLOCK_SIZE
                mask = (j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < reduce_dim_size
                
                # Load input values
                input_vals = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
                
                # Accumulate
                acc += input_vals
            
            # Sum the accumulated values
            result = tl.sum(acc, axis=0)
            
            # Store result
            output_offset = output_idx * stride_output_reduce
            tl.store(output_ptr + output_offset, result)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for sum reduction.
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
        # Ensure tensor is contiguous
        x = x.contiguous()
        
        # Calculate shapes
        input_shape = x.shape
        output_shape = list(input_shape)
        output_shape[self.dim] = 1
        
        # Calculate sizes
        reduce_dim_size = input_shape[self.dim]
        other_dims_size = 1
        for i in range(len(input_shape)):
            if i != self.dim:
                other_dims_size *= input_shape[i]
        
        # Calculate strides
        stride_input_reduce = 1
        stride_output_reduce = 1
        for i in range(self.dim + 1, len(input_shape)):
            stride_input_reduce *= input_shape[i]
        for i in range(self.dim, len(input_shape)):
            stride_output_reduce *= input_shape[i]
        
        # Prepare output tensor
        output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
        
        # Launch kernel
        if reduce_dim_size > 0:
            grid = (
                triton.cdiv(reduce_dim_size, 128),
                triton.cdiv(other_dims_size, 128)
            )
            
            sum_reduce_kernel[grid](
                x,
                output,
                reduce_dim_size,
                other_dims_size,
                stride_input_reduce,
                stride_output_reduce,
                BLOCK_SIZE=128
            )
        
        return output