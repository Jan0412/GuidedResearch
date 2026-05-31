import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_symmetric_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offset pointers for the output tile
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    mask_m = off_m < N
    mask_n = off_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, N, BLOCK_SIZE_K):
        off_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = off_k < N
        
        # Load block of A: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a = tl.load(
            A_ptr + off_m[:, None] * stride_am + off_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Load block of B: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b = tl.load(
            B_ptr + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # Accumulate matrix multiplication
        acc = acc + tl.dot(a, b, out_dtype=tl.float32)
    
    # Store result
    c = acc.to(tl.float32)
    tl.store(
        C_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn,
        c,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_matmul_symmetric(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized matrix multiplication for symmetric matrices using Triton.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    
    # Allocate output tensor
    C = torch.empty_like(A)
    
    # Set block sizes for optimization (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(N, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    matmul_symmetric_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
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
        Performs matrix multiplication using optimized Triton kernel.
        
        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.
            
        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return triton_matmul_symmetric(A, B)