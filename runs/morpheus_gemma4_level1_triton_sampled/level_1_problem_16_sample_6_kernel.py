import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A^T * B.
    A is provided as (K, M), so A^T is (M, K).
    B is provided as (K, N).
    C is (M, N).
    """
    # Map program IDs to the block of C being computed
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # L2 cache optimization: group blocks in M dimension
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m)
    
    # Re-order to improve L2 cache hit rate
    # This is a common optimization for GEMM
    group_id = pid_n // (num_pid_n // GROUP_SIZE_M if num_pid_n >= GROUP_SIZE_M else 1)
    first_pid_m = group_id * GROUP_SIZE_M
    pid_m = (pid_m + first_pid_m) % num_pid_m
    
    # Pointers to the start of the blocks
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers to A and B blocks
    # A is (K, M), but we treat it as A^T (M, K). 
    # So A^T[m, k] = A[k, m].
    # a_ptr + m_offs * stride_am + k_offs * stride_ak
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        # Masking to handle cases where M, N, K are not multiples of BLOCK_SIZE
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (k * BLOCK_SIZE_K + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k * BLOCK_SIZE_K + offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        
        # Compute dot product
        accumulator += tl.dot(a, b)
        
        # Advance pointers to the next block in K dimension
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store the result in C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)

def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    # A: (K, M), B: (K, N)
    K, M = A.shape
    _, N = B.shape
    
    # Ensure tensors are contiguous on GPU
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Strides
    stride_ak = A.stride(0)
    stride_am = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    
    # Tunable parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid: one program for each block of C
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A^T * B) using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication A^T * B.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)