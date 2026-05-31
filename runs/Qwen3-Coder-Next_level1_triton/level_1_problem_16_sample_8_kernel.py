import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_T_ptr,  # Pointer to A^T (transposed A, so actually stored as (M, K))
    B_ptr,    # Pointer to B (K, N)
    C_ptr,    # Pointer to C (M, N)
    M, N, K,  # Dimensions
    stride_am, stride_ak,  # Strides for A^T (row-major: M x K)
    stride_bk, stride_bn,  # Strides for B (row-major: K x N)
    stride_cm, stride_cn,  # Strides for C (row-major: M x N)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create blocks of rows and columns
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouping for better performance (similar to cuBLAS)
    # This helps improve cache locality by having threads work on adjacent rows
    num_blocks_in_group = GROUP_SIZE_M * num_blocks_n
    group_id = pid_m // GROUP_SIZE_M
    first_block_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_blocks_m - first_block_m, GROUP_SIZE_M)
    pid_m = first_block_m + (pid_m % group_size_m)
    
    # Calculate start indices for the blocks
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create ranges for the block indices
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to handle edge cases where M or N aren't multiples of block sizes
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Calculate K offsets
        offset_k = k * BLOCK_SIZE_K
        offsets_k = offset_k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offsets_k < K
        
        # Load block of A^T: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # A^T is stored as (M, K) row-major, so A^T[m, k] = A[k, m]
        a_ptrs = A_T_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load block of B: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # B is stored as (K, N) row-major
        b_ptrs = B_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Matrix multiply: acc += a @ b
        # Using tl.dot with tensor core friendly layout
        acc = tl.dot(a, b, acc)
    
    # Store result to C
    c_ptrs = C_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Convert accumulator to float32 if needed (it's already float32 from initialization)
    tl.store(c_ptrs, acc, mask=mask)


def triton_matmul(A_T: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A^T @ B using Triton kernel.
    
    Args:
        A_T: Transposed A tensor of shape (M, K) [A was originally (K, M)]
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A_T.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A_T = A_T.contiguous()
    B = B.contiguous()
    
    M, K = A_T.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    
    # Allocate output tensor
    C = torch.empty((M, N), dtype=A_T.dtype, device=A_T.device)
    
    # Calculate grid dimensions
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A_T, B, C,
        M, N, K,
        A_T.stride(0), A_T.stride(1),
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
    Optimized model that performs matrix multiplication C = A^T * B using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A^T @ B using optimized Triton kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Transpose A to get (M, K) before passing to Triton kernel
        A_T = A.T
        return triton_matmul(A_T, B)