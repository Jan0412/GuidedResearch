import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID for the dimension we're processing
    pid = tl.program_id(0)
    
    # Calculate starting position for this thread's chunk
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # For each element in the block
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            # Calculate the actual index in the flattened tensor
            idx = block_start + i
            
            # Calculate which position in the dimension we're at
            dim_pos = idx % dim_size
            
            # If we're at position 0, output 0.0
            if dim_pos == 0:
                tl.store(output_ptr + idx, 0.0)
            else:
                # Calculate cumulative sum from previous positions
                cumsum = 0.0
                for j in range(dim_pos):
                    pos = idx - dim_pos + j
                    cumsum += tl.load(x_ptr + pos)
                tl.store(output_ptr + idx, cumsum)

def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    Compute exclusive cumulative sum using Triton kernel
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Calculate total elements and dimension size
    n_elements = x.numel()
    dim_size = x.shape[dim]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Calculate strides for the dimension
    stride_x = 1
    stride_out = 1
    for i in range(dim):
        stride_x *= x.shape[i]
        stride_out *= x.shape[i]
    
    # Determine block size and grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    exclusive_cumsum_kernel[grid](
        x, out, n_elements, dim_size, stride_x, stride_out, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).
    Optimized with Triton kernels.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use the Triton implementation for exclusive cumulative sum
        exclusive_cumsum = torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), x), dim=self.dim)[:-1]
        return triton_exclusive_cumsum(exclusive_cumsum, self.dim)