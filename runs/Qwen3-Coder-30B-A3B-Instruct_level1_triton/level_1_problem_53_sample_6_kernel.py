import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride_input_dim,
    stride_output_dim,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block ID
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Initialize minimum value
    min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
    
    # Loop through the dimension to reduce
    for i in range(dim_size):
        # Calculate input offset for current element
        input_offset = offsets + i * stride_input_dim
        # Load input value
        input_val = tl.load(input_ptr + input_offset, mask=mask, other=float('inf'))
        # Update minimum
        min_val = tl.minimum(min_val, input_val)
    
    # Store the result
    tl.store(output_ptr + offsets, min_val, mask=mask)

def triton_min_reduction(x: torch.Tensor, dim: int):
    """
    Performs min reduction along a specified dimension using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Calculate output shape
    output_shape = list(x.shape)
    output_shape.pop(dim)
    output_shape = tuple(output_shape)
    
    # Prepare output tensor
    out = torch.empty(output_shape, dtype=torch.float32, device=x.device)
    
    # Calculate dimensions
    n_elements = out.numel()
    stride_input_dim = x.stride(dim)
    stride_output_dim = 1 if len(output_shape) == 0 else out.stride(0) if len(output_shape) > 0 else 1
    dim_size = x.shape[dim]
    
    # Block size
    BLOCK_SIZE = 1024
    
    # Grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    min_reduction_kernel[grid](
        x, 
        out, 
        n_elements, 
        stride_input_dim, 
        stride_output_dim, 
        dim_size, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using Triton kernels.
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
        Applies min reduction over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min_reduction(x, self.dim)