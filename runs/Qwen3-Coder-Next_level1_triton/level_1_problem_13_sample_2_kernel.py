import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,  # Matrix dimension (N x N)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create block offsets for M and N dimensions
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create pointers for the blocks of A and B
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_SIZE_K):
        # Load blocks from A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < N - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < N - k, other=0.0)
        
        # Accumulate matrix multiplication
        acc += tl.dot(a, b, out_dtype=tl.float32)
        
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Convert to float16 if needed and store result
    c = acc.to(tl.float32)
    
    # Store the result
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized matrix multiplication using Triton kernel.
    Assumes square matrices of size N x N.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Configure block sizes (tuned for modern GPUs)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(N, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        1
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        N,
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