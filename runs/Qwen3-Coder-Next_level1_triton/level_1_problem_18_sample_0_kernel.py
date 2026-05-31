import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_transposed_kernel(
    A_ptr,  # Input matrix A of shape (K, M) - we'll treat as A^T in computation
    B_ptr,  # Input matrix B of shape (N, K) - we'll treat as B^T in computation
    C_ptr,  # Output matrix of shape (M, N)
    M, N, K,
    stride_am, stride_ak,  # Strides for A
    stride_bk, stride_bn,  # Strides for B
    stride_cm, stride_cn,  # Strides for C
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create ranges for M and N dimensions
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A block (shape: BLOCK_SIZE_M x BLOCK_SIZE_K)
        # A is indexed as A[offsets_m, offsets_k] but we need to handle strides
        a_ptrs = A_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        a_mask = mask_m[:, None] & (offsets_k < K)[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block (shape: BLOCK_SIZE_K x BLOCK_SIZE_N)
        # B is indexed as B[offsets_k, offsets_n]
        b_ptrs = B_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        b_mask = (offsets_k < K)[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiply: accumulator += a @ b
        accumulator = tl.dot(a, b, accumulator)
    
    # Convert accumulator to appropriate dtype if needed
    c = accumulator.to(tl.float32)
    
    # Store result
    c_ptrs = C_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul_transposed(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A^T * B^T using Triton kernel.
    
    Args:
        A: Input tensor of shape (K, M)
        B: Input tensor of shape (N, K)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    K, M = A.shape
    N, _ = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid configuration
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        1
    )
    
    # Launch kernel
    matmul_transposed_kernel[grid](
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
    Optimized model that performs transposed matrix multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A^T * B^T using optimized Triton kernel.
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).
        
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul_transposed(A, B)