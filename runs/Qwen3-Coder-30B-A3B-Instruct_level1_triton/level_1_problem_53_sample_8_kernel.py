import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_reduction_kernel(
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
        # Create offsets for current block
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load input values
        x = tl.load(input_ptr + input_base + offsets * stride_input_dim, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        local_min = tl.minimum(x, tl.broadcast_to(tl.reduce(x, axis=0, combine_fn=tl.minimum), x.shape))
        
        # Store result
        tl.store(output_ptr + output_base + offsets * stride_output_dim, local_min, mask=mask)

def triton_min_reduction(x: torch.Tensor, dim: int):
    """
    Triton-based min reduction implementation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Handle negative dimensions
    if dim < 0:
        dim = x.dim() + dim
    
    # Calculate output shape
    output_shape = list(x.shape)
    output_shape.pop(dim)
    
    # Prepare output tensor
    out = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    # Calculate strides
    stride_input_batch = 1
    stride_input_dim = 1
    stride_output_batch = 1
    stride_output_dim = 1
    
    for i in range(dim + 1, x.dim()):
        stride_input_dim *= x.shape[i]
        stride_output_dim *= out.shape[i - 1] if i > 0 else 1
    
    for i in range(0, dim):
        stride_input_batch *= x.shape[i]
        stride_output_batch *= out.shape[i] if i < len(out.shape) else 1
    
    # Number of elements in the tensor
    n_elements = x.numel()
    dim_size = x.shape[dim]
    
    # Determine block size and grid
    BLOCK_SIZE = 128
    grid = (stride_input_batch,)
    
    # Launch the Triton kernel
    min_reduction_kernel[grid](
        x,
        out,
        n_elements,
        dim_size,
        stride_input_batch,
        stride_input_dim,
        stride_output_batch,
        stride_output_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for min reduction.
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
        Applies min reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min_reduction(x, self.dim)