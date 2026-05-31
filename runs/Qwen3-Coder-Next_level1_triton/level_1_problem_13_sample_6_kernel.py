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
):
    # Get program indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for M and N dimensions
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for valid indices
    mask_m = offsets_m < N
    mask_n = offsets_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_SIZE_K):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offsets_k < N
        
        # Load A block: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a = tl.load(
            A_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Load B block: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b = tl.load(
            B_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # Accumulate matrix multiplication
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Store result
    c = acc.to(tl.float32)
    tl.store(
        C_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn,
        c,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized matrix multiplication for symmetric matrices using Triton.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Input matrices must be square."
    
    N = A.shape[0]
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Define block sizes for optimal performance
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(N, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
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
        return triton_matmul(A, B)