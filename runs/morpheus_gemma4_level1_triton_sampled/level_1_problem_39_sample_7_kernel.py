import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    x_ptr,      # Pointer to input tensor
    out_ptr,    # Pointer to output tensor
    M,          # Number of rows (batch size)
    N,          # Number of columns (dimension)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program (block) handles one row of the input matrix
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    # Calculate the starting pointer for the current row
    row_start_ptr = x_ptr + row_idx * N
    out_start_ptr = out_ptr + row_idx * N

    # Pass 1: Compute the sum of squares for the row
    sum_sq = 0.0
    # We use a fixed range with a break to handle dynamic N
    for i in range(0, 1024): 
        if i >= tl.cdiv(N, BLOCK_SIZE):
            break
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute the inverse of the L2 norm
    # Note: 1.0 / sqrt(0) will result in inf, matching PyTorch's behavior
    inv_norm = 1.0 / tl.sqrt(sum_sq)

    # Pass 2: Normalize the row by multiplying by the inverse norm
    for i in range(0, 1024):
        if i >= tl.cdiv(N, BLOCK_SIZE):
            break
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        out = x * inv_norm
        tl.store(out_start_ptr + offsets, out, mask=mask)

def triton_l2_norm(x: torch.Tensor):
    """
    Triton wrapper for L2 normalization along dim=1.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure tensor is contiguous
    x = x.contiguous()
    M, N = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)

    # Block size for processing the dimension N
    BLOCK_SIZE = 1024
    
    # Grid: one block per row
    grid = (M,)

    # Launch kernel
    l2_norm_kernel[grid](
        x, out, M, N, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor along dim=1.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied.
        """
        return triton_l2_norm(x)