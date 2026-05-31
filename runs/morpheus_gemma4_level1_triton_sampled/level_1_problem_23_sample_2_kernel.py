import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr, 
    out_ptr, 
    dim, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row of the batch
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the current row
    x_row_ptr = x_ptr + row_idx * dim
    out_row_ptr = out_ptr + row_idx * dim

    # --- Pass 1: Find the maximum value in the row for numerical stability ---
    row_max = -float('inf')
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(vals))

    # --- Pass 2: Compute the sum of exponentials ---
    row_sum = 0.0
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
        row_sum += tl.sum(tl.exp(vals - row_max))

    # --- Pass 3: Compute final softmax and store ---
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
        res = tl.exp(vals - row_max) / row_sum
        tl.store(out_row_ptr + offsets, res, mask=mask)

def triton_softmax(x: torch.Tensor):
    """
    Triton wrapper for the softmax operation along the last dimension.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    # Block size for the inner loop
    BLOCK_SIZE = 1024 
    
    # Grid is one program per row in the batch
    grid = (batch_size,)
    
    softmax_kernel[grid](
        x, 
        out, 
        dim, 
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