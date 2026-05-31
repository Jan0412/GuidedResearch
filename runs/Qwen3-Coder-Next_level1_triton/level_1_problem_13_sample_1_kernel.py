import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def symmetric_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_a, stride_b, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    # Block indices for row (m) and column (n) of output matrix C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute starting offsets for rows and columns
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid indices
    mask_m = offs_m < N
    mask_n = offs_n < N
    
    # Initialize accumulator for dot product
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Iterate over the k dimension (columns of A / rows of B)
    for k in range(0, N, BLOCK_SIZE):
        # Compute k offsets
        offs_k = k + tl.arange(0, BLOCK_SIZE)
        mask_k = offs_k < N
        
        # Load A[offs_m, offs_k] with row-major indexing
        a_ptrs = A_ptr + offs_m[:, None] * stride_a + offs_k[None, :]
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load B[offs_k, offs_n] with row-major indexing
        b_ptrs = B_ptr + offs_k[:, None] * stride_b + offs_n[None, :]
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate the matrix multiplication part
        acc = tl.dot(a, b, acc)
    
    # Store the result to C[offs_m, offs_n]
    c_ptrs = C_ptr + offs_m[:, None] * stride_c + offs_n[None, :]
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def triton_symmetric_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication of two symmetric matrices using a custom Triton kernel.
    Despite symmetry, we compute the full matrix product (not exploiting symmetry for optimization
    since the result of multiplying two symmetric matrices is not necessarily symmetric).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    assert len(A.shape) == 2, "Input matrices must be 2D."
    assert A.shape[0] == A.shape[1], "Input matrices must be square."
    
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    C = torch.empty_like(A)
    
    # Grid dimensions: (num_blocks_m, num_blocks_n)
    BLOCK_SIZE = 128
    grid = (
        triton.cdiv(N, BLOCK_SIZE),
        triton.cdiv(N, BLOCK_SIZE)
    )
    
    # Launch kernel
    symmetric_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), B.stride(0), C.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices using Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return triton_symmetric_matmul(A, B)