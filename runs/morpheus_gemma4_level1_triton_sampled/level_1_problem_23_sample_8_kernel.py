import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr,      # Pointer to input tensor
    out_ptr,    # Pointer to output tensor
    batch_size, # Total number of rows
    dim,        # Total number of columns
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    if row_idx >= batch_size:
        return

    # Pointers for the start of the current row
    row_x_ptr = x_ptr + row_idx * dim
    row_out_ptr = out_ptr + row_idx * dim

    # --- Pass 1: Online Softmax to find max and sum of exponentials ---
    # We use the online softmax algorithm to compute the normalization constants in one pass.
    row_max = -float('inf')
    row_sum = 0.0

    for i in range(0, dim, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim
        # Load block of values; use -inf for masked elements to not affect max/sum
        vals = tl.load(row_x_ptr + cols, mask=mask, other=-float('inf'))
        
        # Find the maximum of the current block
        block_max = tl.max(vals)
        
        # Update the global max and the running sum
        # Formula: sum_new = sum_old * exp(max_old - max_new) + sum(exp(vals - max_new))
        new_max = tl.maximum(row_max, block_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(vals - new_max))
        row_max = new_max

    # --- Pass 2: Compute and store the final softmax values ---
    for i in range(0, dim, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim
        vals = tl.load(row_x_ptr + cols, mask=mask, other=-float('inf'))
        
        # Softmax formula: exp(x - max) / sum(exp(x - max))
        out = tl.exp(vals - row_max) / row_sum
        tl.store(row_out_ptr + cols, out, mask=mask)


def triton_softmax(x: torch.Tensor):
    """
    Wrapper for the Triton softmax kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    batch_size, dim = x.shape
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # BLOCK_SIZE is chosen to be a power of 2. 1024 is typically efficient.
    BLOCK_SIZE = 1024
    
    # Launch one program per row
    grid = (batch_size,)
    
    softmax_kernel[grid](
        x, out, batch_size, dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softmax activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using Triton for acceleration.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x)