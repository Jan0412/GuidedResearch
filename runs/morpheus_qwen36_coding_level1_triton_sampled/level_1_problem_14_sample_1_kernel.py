import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triangular_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_a_row, stride_a_col,
    stride_b_row, stride_b_col,
    stride_c_row, stride_c_col,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID maps to block coordinates
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    block_row = pid // num_blocks
    block_col = pid % num_blocks

    # Block start indices
    row_off = block_row * BLOCK_SIZE
    col_off = block_col * BLOCK_SIZE

    # Offsets within the block
    offs_row = row_off + tl.arange(0, BLOCK_SIZE)
    offs_col = col_off + tl.arange(0, BLOCK_SIZE)

    # Global row and column indices
    rows = offs_row[:, None]
    cols = offs_col[None, :]

    # Mask for valid output elements (upper triangular)
    mask = rows < N
    mask &= cols < N
    mask &= rows <= cols  # Only compute upper triangular part

    if not mask.any():
        return

    # Accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Loop over K dimension
    # For upper triangular matmul, k ranges from row to col
    # We iterate over blocks of K
    num_k_blocks = tl.cdiv(N, BLOCK_SIZE)
    for k_block in range(num_k_blocks):
        k_off = k_block * BLOCK_SIZE
        k_offsets = k_off + tl.arange(0, BLOCK_SIZE)

        # Load A tile with mask k >= row
        a_offsets = k_offsets[None, :]
        a_mask = (rows <= a_offsets) & (a_offsets < N) & mask
        a_ptrs = A_ptr + rows * stride_a_row + a_offsets * stride_a_col
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile with mask k <= col
        b_offsets = k_offsets[:, None]
        b_mask = (b_offsets <= cols) & (b_offsets < N) & mask
        b_ptrs = B_ptr + b_offsets * stride_b_row + cols * stride_b_col
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Dot product
        acc += tl.dot(a, b)

    # Store result
    c_ptrs = C_ptr + rows * stride_c_row + cols * stride_c_col
    tl.store(c_ptrs, acc, mask=mask)


def triangular_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A @ B for upper triangular matrices A and B,
    returning only the upper triangular part of the result.
    """
    assert A.is_cuda and B.is_cuda
    assert A.shape == B.shape
    N = A.shape[0]
    
    C = torch.zeros_like(A)
    
    BLOCK_SIZE = 64
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (num_blocks * num_blocks,)
    
    triangular_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triangular_matmul(A, B)