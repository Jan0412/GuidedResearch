import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tril_matmul_kernel(
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
    
    # Skip upper triangular part
    if pid_m < pid_n:
        return
    
    # Create offsets for M and N dimensions
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for valid indices (only lower triangular)
    mask_m = offsets_m < N
    mask_n = offsets_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension with tiling
    for k in range(0, N, BLOCK_SIZE_K):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offsets_k < N
        
        # Load A tiles: A[i,k] where i >= k (but we only need A[i,k] where i >= j and k <= i)
        # For lower triangular A, A[i,k] = 0 if k > i, so we use mask for k <= i
        a_offsets = offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        a_mask = mask_m[:, None] & mask_k[None, :] & (offsets_k[None, :] <= offsets_m[:, None])
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load B tiles: B[k,j] where k >= j (but we only need B[k,j] where k <= j for lower triangular)
        # For lower triangular B, B[k,j] = 0 if k > j, so we use mask for k <= j
        b_offsets = offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        b_mask = mask_k[:, None] & mask_n[None, :] & (offsets_k[:, None] <= offsets_n[None, :])
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Accumulate product
        accumulator += tl.dot(a, b, out_dtype=tl.float32)
    
    # Store result
    c_offsets = offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    c_mask = mask
    tl.store(C_ptr + c_offsets, accumulator, mask=c_mask)


def triton_tril_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized matrix multiplication for lower triangular matrices.
    Only computes the lower triangular part of A @ B.
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
    
    # Tunable parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Grid: one block per tile in the lower triangular region
    grid = (
        triton.cdiv(N, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    tril_matmul_kernel[grid](
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
    Optimized model that performs matrix multiplication of lower triangular matrices
    using a custom Triton kernel that only computes the lower triangular part.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The lower triangular part of A @ B.
        """
        return triton_tril_matmul(A, B)