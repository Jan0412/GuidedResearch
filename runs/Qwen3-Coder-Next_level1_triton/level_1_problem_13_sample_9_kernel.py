import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Matrix multiplication: C = A @ B
    # M = number of rows in A and C
    # N = number of columns in B and C
    # K = number of columns in A and rows in B
    
    # Get program ID and compute tile indices
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create offset indices for the tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointers for A and B
    A_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B tiles
        a = tl.load(A_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(B_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Accumulate the matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
        
        # Update pointers for next iteration
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Convert accumulator to float16 if needed and store result
    C_ = accumulator.to(tl.float32)
    
    # Write back the result
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, C_, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication C = A @ B using Triton kernel.
    Optimized for FP32 precision.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Incompatible dimensions for matrix multiplication"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes (tunable parameters for optimization)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of two matrices using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N).
            B (torch.Tensor): Input matrix B, shape (N, N).

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return triton_matmul(A, B)