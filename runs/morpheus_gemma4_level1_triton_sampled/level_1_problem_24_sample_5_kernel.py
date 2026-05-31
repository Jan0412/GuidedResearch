import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    x_ptr, 
    out_ptr, 
    dim, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch element)
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the row for input and output
    x_ptr_row = x_ptr + row_idx * dim
    out_ptr_row = out_ptr + row_idx * dim

    # Pass 1: Find the maximum value in the row for numerical stability
    max_val = -float('inf')
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(x_ptr_row + offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(vals))

    # Pass 2: Compute the sum of exponentials (x - max_val)
    sum_exp = 0.0
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(x_ptr_row + offsets, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(vals - max_val))

    # Compute Log-Sum-Exp
    lse = max_val + tl.log(sum_exp)

    # Pass 3: Compute the final LogSoftmax (x - lse) and store
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(x_ptr_row + offsets, mask=mask, other=-float('inf'))
        tl.store(out_ptr_row + offsets, vals - lse, mask=mask)

def triton_log_softmax(x: torch.Tensor):
    """
    Triton wrapper for log_softmax.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure input is contiguous for pointer arithmetic
    x = x.contiguous()
    batch_size, dim = x.shape
    
    out = torch.empty_like(x)
    
    # Block size for the inner loops
    BLOCK_SIZE = 1024
    
    # Grid is one program per row
    grid = (batch_size,)
    
    log_softmax_kernel[grid](
        x, out, dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a LogSoftmax activation using a custom Triton kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # The original model used dim=1 for reduction.
        # Our Triton kernel is optimized for reduction over the last dimension.
        return triton_log_softmax(x)