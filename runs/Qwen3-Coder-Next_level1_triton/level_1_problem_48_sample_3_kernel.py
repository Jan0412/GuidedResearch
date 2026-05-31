import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    reduce_size,  # Size of the dimension being reduced
    other_size,  # Product of sizes of other dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one element in the output (one mean calculation)
    # Output has shape [batch_size, other_size] after reduction
    batch_idx = tl.program_id(0) // other_size
    other_idx = tl.program_id(0) % other_size
    
    # Calculate the starting position in the input tensor
    # Input shape is [batch_size, reduce_size, other_size] conceptually
    # Memory layout is contiguous, so we need to compute the offset
    base_offset = batch_idx * (reduce_size * other_size) + other_idx
    
    # Accumulator for the sum
    sum = tl.zeros([1], dtype=tl.float32)
    
    # Process the reduction dimension in chunks
    for start in range(0, reduce_size, BLOCK_SIZE):
        offset = start * other_size
        ptr = x_ptr + base_offset + offset
        
        # Load data - need to handle case where reduce_size is not divisible by BLOCK_SIZE
        mask = tl.arange(0, BLOCK_SIZE) < (reduce_size - start)
        # Load multiple values along the reduction dimension
        x = tl.load(ptr, mask=mask, other=0.0)
        sum += tl.sum(x, axis=0)
    
    # Compute mean
    mean_val = sum / reduce_size
    
    # Store result
    tl.store(out_ptr + tl.program_id(0), mean_val)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute mean along specified dimension using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get input shape
    shape = x.shape
    ndim = len(shape)
    
    # Normalize dimension
    if dim < 0:
        dim += ndim
    
    # Calculate output shape
    output_shape = list(shape)
    del output_shape[dim]
    output_shape = tuple(output_shape)
    
    # Calculate dimensions for kernel
    batch_size = 1
    for i in range(dim):
        batch_size *= shape[i]
    
    reduce_size = shape[dim]
    
    other_size = 1
    for i in range(dim + 1, len(shape)):
        other_size *= shape[i]
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate grid size
    total_elements = batch_size * other_size
    BLOCK_SIZE = 256  # Tunable parameter
    
    # Launch kernel
    grid = (total_elements,)
    mean_kernel[grid](x, out, batch_size, reduce_size, other_size, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using Triton kernel.
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
        Reduces the input tensor along the specified dimension by taking the mean using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        return triton_mean(x, self.dim)