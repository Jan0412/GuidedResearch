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
    Custom Triton kernel for matrix multiplication.
    C[M, N] = A[M, K] @ B[K, N]
    """
    # Map program IDs to the block of the output matrix C
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # L2 cache optimization: group programs to improve data reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) // num_pid_n
    pid_n = (pid % num_pid_in_group) % num_pid_n

    # Pointers to the start of the blocks in A, B, and C
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Iterate over the K dimension to compute the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks from A and B
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (k * BLOCK_SIZE_K + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k * BLOCK_SIZE_K + offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        
        # Perform matrix multiplication for the block
        accumulator += tl.dot(a, b)
        
        # Advance pointers to the next block along the K dimension
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store the result back to C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel.
    Handles reshaping of the 4D tensor A and the 2D tensor B.
    """
    # A shape: (b, i, j, l) -> flattened to (M, K) where M = b*i*j, K = l
    # B shape: (l, k) -> (K, N) where N = k
    b, i, j, l = A.shape
    k_dim = B.shape[1]
    
    M = b * i * j
    N = k_dim
    K = l

    # Ensure tensors are contiguous on GPU
    A = A.contiguous().view(M, K)
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Grid configuration
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    # Reshape output back to (b, i, j, k)
    return C.view(b, i, j, k_dim)


class ModelNew(nn.Module):
    """
    Optimized version of the 4D tensor-matrix multiplication using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication.
        A: (b, i, j, l)
        B: (l, k)
        Returns: (b, i, j, k)
        """
        return triton_matmul(A, B)