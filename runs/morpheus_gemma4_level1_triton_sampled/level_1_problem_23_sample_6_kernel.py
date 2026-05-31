import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr, 
    out_ptr, 
    stride_row, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_id = tl.program_id(0)
    row_ptr = x_ptr + row_id * stride_row

    # 1. Find the maximum value in the row for numerical stability
    row_max = -float('inf')
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(row_ptr + offsets, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))

    # 2. Compute the sum of exponentials (shifted by max)
    row_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(row_ptr + offsets, mask=mask, other=-float('inf'))
        row_sum += tl.sum(tl.exp(vals - row_max), axis=0)

    # 3. Compute the final softmax values and store them
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(row_ptr + offsets, mask=mask, other=-float('inf'))
        out = tl.exp(vals - row_max) / row_sum
        tl.store(out_ptr + offsets, out, mask=mask)

def triton_softmax(x: torch.Tensor):
    """
    Wrapper for the Triton softmax kernel.
    """
    # Softmax is typically computed on GPU
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure the tensor is contiguous for pointer arithmetic
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    stride_row = x.stride(0)
    
    # Block size for the internal loops
    BLOCK_SIZE = 1024
    
    # Launch one program per row
    grid = (n_rows,)
    
    softmax_kernel[grid](
        x, 
        out, 
        stride_row, 
        n_cols, 
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
        Applies Softmax activation to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x)