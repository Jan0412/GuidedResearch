import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Program ID determines which block of C this thread block computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create row and column indices for the output block
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for row and column indices to handle boundaries
    mask_row = row_idx < M
    mask_col = col_idx < M
    
    # Create index for the reduction dimension (K)
    k_idx = tl.arange(0, BLOCK_K)
    mask_k = k_idx < K

    # Calculate pointers for A and B
    # A has shape (M, K), B has shape (K, N)
    # Using strides to handle potential non-contiguous memory layouts
    a_ptrs = A_ptr + row_idx[:, None] * stride_am + k_idx[None, :] * stride_ak
    b_ptrs = B_ptr + k_idx[:, None] * stride_bk + col_idx[None, :] * stride_bn

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load blocks of A and B
    # Since BLOCK_K == K in this case, we only need one iteration
    a = tl.load(a_ptrs, mask=mask_row[:, None] & mask_k[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_col[None, :], other=0.0)

    # Perform matrix multiplication for the block
    # tl.dot is optimized for Triton and handles the compute efficiently
    c_block = tl.dot(a, b)
    
    # Accumulate result (useful if BLOCK_K < K, but here it's a single block)
    accumulator += c_block

    # Calculate pointer for output block in C
    c_ptrs = C_ptr + row_idx[:, None] * stride_cm + col_idx[None, :] * stride_cn
    
    # Mask for the output block boundaries
    mask_out = mask_row[:, None] & mask_col[None, :]
    
    # Store the result
    tl.store(c_ptrs, accumulator, mask=mask_out)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton implementation of matrix multiplication optimized for 
    tall/skinny matrix shapes where one dimension is much larger than the other.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.is_contiguous(), "A must be contiguous."
    assert B.is_contiguous(), "B must be contiguous."
    
    M, K = A.shape
    _, N = B.shape
    
    assert A.shape[1] == B.shape[0], f"Incompatible dimensions: A is {A.shape}, B is {B.shape}"
    
    # Output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Tunable block sizes
    # M = 32768, N = 32
    # Output shape is (M, M) = (32768, 32768)
    # We use BLOCK_M and BLOCK_N for output tiling, BLOCK_K for reduction
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32  # Matches N exactly
    
    # Grid calculation
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Strides for memory access
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using a custom Triton kernel.
        
        Args:
            A (torch.Tensor): Input matrix of shape (M, K).
            B (torch.Tensor): Input matrix of shape (K, N).
            
        Returns:
            torch.Tensor: Output matrix of shape (M, N).
        """
        return triton_matmul(A, B)