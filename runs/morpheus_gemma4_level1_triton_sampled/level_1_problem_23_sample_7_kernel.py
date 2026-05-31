import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr, 
    out_ptr, 
    batch_size, 
    dim, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the batch
    row_id = tl.program_id(0)
    if row_id >= batch_size:
        return

    # Pointers for the current row
    row_x_ptr = x_ptr + row_id * dim
    row_out_ptr = out_ptr + row_id * dim

    # --- Pass 1: Find Max ---
    row_max = -float('inf')
    for i in range(0, dim, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim
        vals = tl.load(row_x_ptr + cols, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))

    # --- Pass 2: Compute Sum of Exponentials ---
    row_sum = 0.0
    for i in range(0, dim, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim
        vals = tl.load(row_x_ptr + cols, mask=mask, other=-float('inf'))
        exp_vals = tl.exp(vals - row_max)
        # Mask out-of-bounds elements to 0 for the sum
        exp_vals = tl.where(mask, exp_vals, 0.0)
        row_sum += tl.sum(exp_vals, axis=0)

    # --- Pass 3: Compute and Store Softmax ---
    for i in range(0, dim, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim
        vals = tl.load(row_x_ptr + cols, mask=mask, other=-float('inf'))
        out = tl.exp(vals - row_max) / row_sum
        tl.store(row_out_ptr + cols, out, mask=mask)

def triton_softmax(x: torch.Tensor):
    """
    Triton wrapper for softmax along dim=1.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)

    # BLOCK_SIZE must be a power of 2. 1024 is generally a good balance.
    BLOCK_SIZE = 1024
    
    # Grid: one program per row
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
        Applies Softmax activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x)