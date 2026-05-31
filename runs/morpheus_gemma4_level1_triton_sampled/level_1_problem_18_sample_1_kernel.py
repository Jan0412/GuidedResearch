import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, # Pointer to A (K, M)
    b_ptr, # Pointer to B (N, K)
    c_ptr, # Pointer to C (M, N)
    M, N, K,
    stride_am, stride_ak, # Strides for A: (M, 1)
    stride_bn, stride_bk, # Strides for B: (K, 1)
    stride_cm, stride_cn, # Strides for C: (N, 1)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Range of indices for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks in A and B
    # We want a_tile to be (BLOCK_SIZE_M, BLOCK_SIZE_K) and b_tile to be (BLOCK_SIZE_K, BLOCK_SIZE_N)
    # A is (K, M), B is (N, K). 
    # C[i, j] = sum_{k} A[k, i] * B[j, k]
    
    # A pointer: A[k, i] -> a_ptr + k * stride_am + i * stride_ak
    # B pointer: B[j, k] -> b_ptr + j * stride_bn + k * stride_bk
    
    a_ptr_base = a_ptr + rm[:, None] * stride_ak + rk[None, :] * stride_am
    b_ptr_base = b_ptr + rn[None, :] * stride_bn + rk[:, None] * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles
        # Masking for K dimension
        k_offsets = k * BLOCK_SIZE_K + rk
        mask_k = k_offsets < K
        
        # We need to ensure we don't go out of bounds for M and N as well
        mask_m = rm[:, None] < M
        mask_n = rn[None, :] < N
        
        a_tile = tl.load(a_ptr_base + (k * BLOCK_SIZE_K) * stride_am, mask=mask_m & mask_k[None, :], other=0.0)
        b_tile = tl.load(b_ptr_base + (k * BLOCK_SIZE_K) * stride_bk, mask=mask_n & mask_k[:, None], other=0.0)
        
        # Dot product
        accumulator += tl.dot(a_tile, b_tile)

    # Store the result
    c_ptr_final = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptr_final, accumulator, mask=mask_m & mask_n)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    # A: (K, M), B: (N, K)
    K, M = A.shape
    N, _ = B.shape
    
    # Ensure tensors are contiguous and on GPU
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    
    # Output C: (M, N)
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)
    
    # Strides
    stride_am, stride_ak = M, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_cn = N, 1
    
    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A^T * B^T)
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication via Triton kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)