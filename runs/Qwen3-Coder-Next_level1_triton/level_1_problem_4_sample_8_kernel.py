import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk, stride_bh,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID is the row of the matrix C to compute
    pid = tl.program_id(0)
    
    # Create offset for the row of matrix A and C
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < M
    
    # Create offset for the column of matrix B (only one column since it's a vector)
    col_offset = 0  # Only one column in B and C
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = (
            row_offsets[:, None] * stride_am +
            k_offsets[None, :] * stride_ak
        )
        a = tl.load(A_ptr + a_offsets, mask=(row_mask[:, None] & k_mask[None, :]), other=0.0)
        
        # Load B block: shape (BLOCK_SIZE_K, 1)
        b_offsets = (
            k_offsets[:, None] * stride_bk +
            col_offset * stride_bh
        )
        b = tl.load(B_ptr + b_offsets, mask=(k_mask[:, None]), other=0.0)
        
        # Accumulate matrix-vector multiplication result
        acc += tl.sum(a * b, axis=1)
    
    # Write result to C
    c_offsets = (
        row_offsets * stride_cm +
        col_offset * stride_cn
    )
    tl.store(C_ptr + c_offsets, acc, mask=row_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector multiplication using Triton kernel.
    
    A: torch.Tensor of shape (M, K)
    B: torch.Tensor of shape (K, 1)
    Returns: torch.Tensor of shape (M, 1)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    
    assert K_b == K, f"Matrix dimensions must match: A.shape={A.shape}, B.shape={B.shape}"
    assert N == 1, "B must be a column vector"
    
    # Prepare output tensor
    C = torch.empty((M, 1), dtype=A.dtype, device=A.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 256
    
    # Compute grid size (number of blocks in M dimension)
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized Triton kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matmul(A, B)