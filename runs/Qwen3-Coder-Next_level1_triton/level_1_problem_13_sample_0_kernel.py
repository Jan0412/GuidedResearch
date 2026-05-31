import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Group ID and local PID
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block offsets
    block_m = pid_m * BLOCK_SIZE_M
    block_n = pid_n * BLOCK_SIZE_N
    
    # Create pointers for the first block of A and B
    offsets_am = block_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = block_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + (offsets_am[:, None] * stride_am + offsets_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offsets_k[:, None] * stride_bk + offsets_bn[None, :] * stride_bn)
    
    # Accumulator for the result
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B blocks
        a_mask = (offsets_am[:, None] < M) & (offsets_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offsets_k[:, None] < K - k * BLOCK_SIZE_K) & (offsets_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
        
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Store result in C
    c = accumulator.to(tl.float32)
    offsets_cm = block_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = block_n + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn)
    c_mask = (offsets_cm[:, None] < M) & (offsets_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized matrix multiplication kernel for symmetric matrices.
    Uses tiled matrix multiplication with tiling parameters tuned for FP32.
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Check dimensions
    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert B.shape[0] == B.shape[1], "Matrix B must be square"
    assert A.shape[0] == B.shape[0], "Matrices must have same dimensions for multiplication"
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Matrices must be compatible for multiplication"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes (tuned for FP32 on modern GPUs)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid dimensions
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
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using a custom Triton kernel.
    This implementation is optimized for symmetric matrices but works for any square matrices.
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


# Override the original N for consistency
N = 4096

def get_inputs():
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric tensors A and B.
    """
    A = torch.rand(N, N)
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.rand(N, N)
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []