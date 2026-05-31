import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduction_kernel(
    input_ptr,
    output_ptr,
    stride_input_batch,
    stride_input_dim,
    stride_input_other,
    stride_output_batch,
    stride_output_other,
    batch_size,
    reduce_dim_size,
    other_dims_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Get the other dimensions index
    other_idx = tl.program_id(1)
    
    # Calculate the starting position in the input tensor
    input_base_offset = batch_idx * stride_input_batch + other_idx * stride_input_other
    
    # Calculate the starting position in the output tensor
    output_base_offset = batch_idx * stride_output_batch + other_idx * stride_output_other
    
    # Shared memory for reduction
    shared_mem = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over the reduction dimension
    for i in range(0, reduce_dim_size, BLOCK_SIZE):
        # Calculate the actual offset in the input tensor
        offset = input_base_offset + i * stride_input_dim
        
        # Load data into shared memory
        mask = (i + tl.arange(0, BLOCK_SIZE)) < reduce_dim_size
        input_vals = tl.load(input_ptr + offset, mask=mask, other=0.0)
        
        # Store in shared memory
        tl.store(shared_mem + tl.arange(0, BLOCK_SIZE), input_vals, mask=mask)
        
        # Synchronize threads to ensure all data is loaded
        tl.sync()
        
        # Perform reduction within block
        for j in range(BLOCK_SIZE):
            if i + j < reduce_dim_size:
                acc += shared_mem[j]
    
    # Store the result
    tl.store(output_ptr + output_base_offset, acc)

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
        # Ensure input is on GPU and contiguous
        x = x.contiguous().to(torch.float32)
        
        # Calculate shapes
        shape = list(x.shape)
        batch_size = 1
        other_dims_size = 1
        reduce_dim_size = shape[self.dim]
        
        # Calculate batch size (product of all dimensions before reduce_dim)
        for i in range(self.dim):
            batch_size *= shape[i]
            
        # Calculate other dimensions size (product of all dimensions after reduce_dim)
        for i in range(self.dim + 1, len(shape)):
            other_dims_size *= shape[i]
            
        # Calculate strides
        stride_input_batch = 1
        stride_input_dim = 1
        stride_input_other = 1
        
        # Calculate strides for input tensor
        for i in range(self.dim + 1, len(shape)):
            stride_input_other *= shape[i]
            
        for i in range(self.dim):
            stride_input_batch *= shape[i]
            
        stride_input_dim = stride_input_batch * shape[self.dim] // batch_size
        
        # Calculate output shape
        output_shape = list(shape)
        output_shape[self.dim] = 1
        
        # Prepare output tensor
        output = torch.empty(output_shape, dtype=torch.float32, device=x.device)
        
        # Calculate strides for output tensor
        stride_output_batch = 1
        stride_output_other = 1
        
        for i in range(self.dim + 1, len(output_shape)):
            stride_output_other *= output_shape[i]
            
        for i in range(self.dim):
            stride_output_batch *= output_shape[i]
            
        # Launch kernel
        if batch_size > 0 and other_dims_size > 0:
            grid = (batch_size, other_dims_size)
            BLOCK_SIZE = 1024
            
            sum_reduction_kernel[grid](
                x,
                output,
                stride_input_batch,
                stride_input_dim,
                stride_input_other,
                stride_output_batch,
                stride_output_other,
                batch_size,
                reduce_dim_size,
                other_dims_size,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
        return output