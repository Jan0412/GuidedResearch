import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triu_matmul_kernel(
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
    
    # Create offsets for rows of A and C (row_m)
    row_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Create offsets for columns of B and C (col_n)
    col_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create offsets for columns of A and rows of B (col_k)
    col_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for A (row_m x col_k)
    mask_am = row_m[:, None] < N
    mask_ak = col_k[None, :] < N
    mask_a = mask_am & mask_ak
    
    # Create masks for B (col_k x col_n)
    mask_bk = col_k[:, None] < N
    mask_bn = col_n[None, :] < N
    mask_b = mask_bk & mask_bn
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        # Load A block: row_m x k_start+col_k
        a_offsets = (row_m[:, None] * stride_am + 
                    (k_start + col_k[None, :]) * stride_ak)
        a = tl.load(A_ptr + a_offsets, mask=mask_a, other=0.0)
        
        # Load B block: k_start+col_k x col_n
        b_offsets = ((k_start + col_k[:, None]) * stride_bk + 
                    col_n[None, :] * stride_bn)
        b = tl.load(B_ptr + b_offsets, mask=mask_b, other=0.0)
        
        # Accumulate product
        accumulator += tl.dot(a, b)
    
    # Cast to output type
    c = accumulator.to(tl.float32)
    
    # Compute output offsets
    c_offsets = (row_m[:, None] * stride_cm + 
                col_n[None, :] * stride_cn)
    
    # Compute upper triangular mask: only keep elements where row_m <= col_n
    mask_cm = row_m[:, None] < N
    mask_cn = col_n[None, :] < N
    mask_tri = mask_cm & mask_cn & (row_m[:, None] <= col_n[None, :])
    
    # Store result only for upper triangular part
    tl.store(C_ptr + c_offsets, c, mask=mask_tri)


def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication for upper triangular matrices using Triton.
    Only computes the upper triangular part of the result.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    N = A.shape[0]
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Define block sizes for tiling
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    # Grid size
    grid = (tl.cdiv(N, BLOCK_SIZE_M), tl.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    triu_matmul_kernel[grid](
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
    Optimized model that performs matrix multiplication for upper triangular matrices
    using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using Triton kernel.
        
        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).
        
        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_triu_matmul(A, B)