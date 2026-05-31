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
    # Matrix multiplication kernel optimized for square matrices
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create offset blocks for matrix C
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Ensure offsets are within bounds
    off_m = tl.where(off_m < M, off_m, M - 1)
    off_n = tl.where(off_n < N, off_n, N - 1)
    
    # Create masks
    mask_m = off_m < M
    mask_n = off_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        off_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        off_k = tl.where(off_k < K, off_k, K - 1)
        mask_k = off_k < K
        
        # Load tiles from A and B
        a_ptrs = A + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
        b_ptrs = B + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn
        
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate multiplication
        acc = tl.dot(a, b, acc)
    
    # Convert accumulator to float16 if needed
    c = acc.to(tl.float32)
    
    # Store result
    c_ptrs = C + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using Triton kernel.
    Optimized for square matrices (N x N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    _, N = B.shape
    
    # Output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid configuration
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
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
    Optimized model that performs matrix multiplication using Triton kernel.
    Takes two symmetric matrices as input and returns their product.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return triton_matmul(A, B)