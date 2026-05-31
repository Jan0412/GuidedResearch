import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride_input_dim,
    stride_output_dim,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Initialize maximum value
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
    
    # Perform reduction within the block
    for i in range(0, dim_size):
        # Calculate the actual index in the flattened array
        idx = offsets + i * stride_input_dim
        # Load current value
        val = tl.load(input_ptr + idx, mask=idx < n_elements, other=-float('inf'))
        # Update maximum
        max_val = tl.maximum(max_val, val)
    
    # Store the result
    tl.store(output_ptr + offsets, max_val, mask=mask)

def triton_max_reduction(x: torch.Tensor, dim: int):
    """
    Performs max reduction along a specific dimension using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    dim_size = shape[dim]
    
    # Calculate total elements and stride for the reduced dimension
    total_elements = 1
    stride_input_dim = 1
    stride_output_dim = 1
    
    for i in range(len(shape)):
        if i < dim:
            stride_input_dim *= shape[i]
        elif i > dim:
            stride_output_dim *= shape[i]
        total_elements *= shape[i]
    
    # Calculate output shape
    output_shape = list(shape)
    output_shape.pop(dim)
    
    # Prepare output tensor
    out = torch.empty(output_shape, dtype=torch.float32, device='cuda')
    
    # Number of elements in the output tensor
    n_elements = out.numel()
    
    # Tunable parameters
    BLOCK_SIZE = 1024
    
    # Grid calculation
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch the kernel
    max_reduction_kernel[grid](
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
        return triton_max_reduction(x, self.dim)