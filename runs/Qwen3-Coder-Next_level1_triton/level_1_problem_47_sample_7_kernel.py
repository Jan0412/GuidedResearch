import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    inner_size,  # Size of the dimension being reduced
    total_elements,  # Total number of elements in the output tensor
    BLOCK_SIZE: tl.constexpr,
    INNER_BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element (one reduced row)
    batch_idx = tl.program_id(0)
    
    # Calculate starting offset for this batch
    start_offset = batch_idx * inner_size
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over the reduction dimension in blocks
    for start in range(0, inner_size, INNER_BLOCK_SIZE):
        offsets = start + tl.arange(0, INNER_BLOCK_SIZE)
        mask = offsets < inner_size
        
        # Load data
        x_offsets = start_offset + offsets
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Accumulate
        acc += tl.sum(x, axis=0)
    
    # Store result
    out_offsets = batch_idx
    tl.store(out_ptr + out_offsets, acc)


def triton_sum_reduction(x: torch.Tensor, dim: int):
    """
    Performs sum reduction over the specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Tensor with the specified dimension reduced via sum
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    ndim = len(shape)
    
    # Normalize dimension
    if dim < 0:
        dim += ndim
    
    # Calculate batch_size (product of all dimensions before dim) and inner_size (dim dimension)
    batch_size = 1
    for i in range(dim):
        batch_size *= shape[i]
    
    inner_size = shape[dim]
    
    # Output shape: same as input but with dim set to 1
    output_shape = list(shape)
    output_shape[dim] = 1
    
    # Prepare output tensor
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate total elements in output
    total_elements = out.numel()
    
    # Set block sizes for optimization
    BLOCK_SIZE = 128
    INNER_BLOCK_SIZE = 256
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Launch the kernel
    sum_reduction_kernel[grid](
        x, out, batch_size, inner_size, total_elements,
        BLOCK_SIZE=BLOCK_SIZE, INNER_BLOCK_SIZE=INNER_BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension using Triton kernel.
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
        Applies sum reduction over the specified dimension using optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return triton_sum_reduction(x, self.dim)