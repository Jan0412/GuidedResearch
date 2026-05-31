import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_AM, stride_AK,
    stride_BK, stride_CN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Block coordinates
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)
    
    # Offsets for rows and columns
    row_offsets = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    # Masks for boundary checking
    row_mask = row_offsets < M
    col_mask = col_offsets < N
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Mask for K dimension
        k_mask = k_offsets < (K - k)
        
        # Load A block
        A_offsets = (row_offsets[:, None] * stride_AM + k_offsets[None, :] * stride_AK)
        A_block = tl.load(A_ptr + A_offsets, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load B block
        B_offsets = (k_offsets[:, None] * stride_BK + col_offsets[None, :] * stride_CN)
        B_block = tl.load(B_ptr + B_offsets, mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(A_block, B_block)
        
    # Store result
    C_offsets = (row_offsets[:, None] * stride_AM + col_offsets[None, :] * stride_CN)
    tl.store(C_ptr + C_offsets, acc, mask=row_mask[:, None] & col_mask[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B using a custom Triton kernel.
    A is reshaped to (-1, l), B is (l, k), result is (b*i*j, k).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    b, i, j, l = A.shape
    k = B.shape[1]
    
    M = b * i * j
    N = k
    K = l
    
    # Reshape A to 2D matrix
    A_flat = A.reshape(M, K)
    
    # Prepare output tensor
    C_flat = torch.empty(M, N, dtype=A.dtype, device=A.device)
    
    # Strides for flattened tensors
    stride_AM = K
    stride_AK = 1
    stride_BK = N
    stride_CN = 1
    
    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Grid calculation
    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A_flat, B, C_flat,
        M, N, K,
        stride_AM, stride_AK,
        stride_BK, stride_CN,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    # Reshape result back to 4D
    return C_flat.reshape(b, i, j, k)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)