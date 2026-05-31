import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    a_ptr,  # Pointer to diagonal vector A (1D, shape: N)
    b_ptr,  # Pointer to matrix B (2D, shape: N x M)
    out_ptr,  # Pointer to output matrix C (2D, shape: N x M)
    N, M,  # Dimensions
    stride_b0, stride_b1,  # Strides for B
    stride_out0, stride_out1,  # Strides for output
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Block row index (corresponds to rows of diagonal matrix)
    pid_n = tl.program_id(0)
    # Block column index (corresponds to columns of B)
    pid_m = tl.program_id(1)
    
    # Offsets for rows of A and B
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Offsets for columns of B and output
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Create masks to ensure indices are within bounds
    n_mask = n_offsets < N
    m_mask = m_offsets < M
    
    # Load diagonal element from A (only one element per row block)
    # Since A is 1D, we only need the n_offsets
    a_val = tl.load(a_ptr + n_offsets, mask=n_mask, other=0.0)
    
    # Load B block: shape (BLOCK_SIZE_N, BLOCK_SIZE_M)
    # B is stored in row-major, so row index is n_offsets[:, None] and column index is m_offsets[None, :]
    b_ptrs = b_ptr + n_offsets[:, None] * stride_b0 + m_offsets[None, :] * stride_b1
    b = tl.load(b_ptrs, mask=n_mask[:, None] & m_mask[None, :], other=0.0)
    
    # Compute element-wise multiplication: diag(A) * B = A[i] * B[i, j]
    # a_val has shape (BLOCK_SIZE_N,) and b has shape (BLOCK_SIZE_N, BLOCK_SIZE_M)
    # Broadcasting will handle the multiplication
    out = a_val[:, None] * b
    
    # Store result
    out_ptrs = out_ptr + n_offsets[:, None] * stride_out0 + m_offsets[None, :] * stride_out1
    tl.store(out_ptrs, out, mask=n_mask[:, None] & m_mask[None, :])


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs C = diag(A) @ B using a custom Triton kernel.
    
    Args:
        A (torch.Tensor): Diagonal vector of shape (N,)
        B (torch.Tensor): Matrix of shape (N, M)
        
    Returns:
        torch.Tensor: Result of shape (N, M)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[0] == B.shape[0], "A.shape[0] must equal B.shape[0]"
    
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    out = torch.empty_like(B)
    
    # Define block sizes for tiling
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 64
    
    # Grid: one block per tile in N and M dimensions
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(M, BLOCK_SIZE_M))
    
    # Launch kernel
    diag_matmul_kernel[grid](
        A, B, out,
        N, M,
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for diag(A) @ B.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)