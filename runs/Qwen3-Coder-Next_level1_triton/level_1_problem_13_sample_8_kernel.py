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
    BLOCK_SIZE: tl.constexpr,
    ACCUMULATOR_dtype: tl.constexpr = tl.float32
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create block offsets for matrix C
    offset_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for bounds checking
    mask_m = offset_m < N
    mask_n = offset_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=ACCUMULATOR_dtype)
    
    # Loop over K dimension in blocks
    for k in range(0, N, BLOCK_SIZE):
        # Compute offsets for A and B
        offset_k = k + tl.arange(0, BLOCK_SIZE)
        mask_k = offset_k < N
        
        # Load blocks from A and B
        # A is row-major, so we load A[i, k] where i is row and k is column
        a_block = tl.load(
            A_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # B is row-major, so we load B[k, j] where k is row and j is column
        b_block = tl.load(
            B_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # Accumulate matrix multiplication
        accumulator = tl.dot(a_block, b_block, accumulator)
    
    # Store result to C
    c_block = accumulator.to(C_ptr.type.element_ty)
    tl.store(
        C_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn,
        c_block,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_matmul_symmetric(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized matrix multiplication for symmetric matrices using Triton.
    
    Args:
        A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
        B (torch.Tensor): Input matrix B, shape (N, N), symmetric.
    
    Returns:
        torch.Tensor: Output matrix C, shape (N, N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Input matrices must be square."
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Block size for tiling - can be tuned
    BLOCK_SIZE = 128
    
    # Grid dimensions for 2D tiling
    grid = (
        triton.cdiv(N, BLOCK_SIZE),
        triton.cdiv(N, BLOCK_SIZE),
    )
    
    # Launch kernel
    matmul_symmetric_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model for symmetric matrix multiplication using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication of two symmetric matrices.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return triton_matmul_symmetric(A, B)