import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_3d_2d_kernel(
    A_ptr,  # Pointer to 3D tensor A of shape (N, M, K)
    B_ptr,  # Pointer to matrix B of shape (K, L)
    C_ptr,  # Pointer to output tensor of shape (N, M, L)
    N, M, K, L,
    stride_a0, stride_a1, stride_a2,  # Strides for A: (M*K, K, 1)
    stride_b0, stride_b1,  # Strides for B: (L, 1)
    stride_c0, stride_c1, stride_c2,  # Strides for C: (M*L, L, 1)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs for batch dimensions
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Calculate base pointers for this batch element
    # Each batch element (n, m) computes a 1×L vector from K×L matrix B
    # A is a 1×K vector at position (n, m, :)
    
    # Offset for A: n*M*K + m*K
    a_offset = pid_n * stride_a0 + pid_m * stride_a1
    
    # Offset for C: n*M*L + m*L
    c_offset = pid_n * stride_c0 + pid_m * stride_c1
    
    # Initialize accumulator for L elements
    accumulator = tl.zeros((BLOCK_SIZE_L,), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        # Load A: 1×K vector
        a_vals = tl.load(A_ptr + a_offset + k_offsets * stride_a2, mask=k_mask, other=0.0)
        
        # Load B: K×L matrix in blocks
        b_offsets = k_offsets[:, None] * stride_b0 + tl.arange(0, BLOCK_SIZE_L)[None, :] * stride_b1
        b_mask = (k_offsets[:, None] < K) & (tl.arange(0, BLOCK_SIZE_L)[None, :] < L)
        b_vals = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Compute partial matmul: (1×K) × (K×L) = (1×L)
        accumulator += tl.sum(a_vals[:, None] * b_vals, axis=0)
    
    # Store result
    c_offsets = tl.arange(0, BLOCK_SIZE_L)
    c_mask = c_offsets < L
    tl.store(C_ptr + c_offset + c_offsets * stride_c2, accumulator, mask=c_mask)


def triton_matmul_3d_2d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 3D tensor-matrix multiplication: A (N, M, K) @ B (K, L) -> C (N, M, L)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    K2, L = B.shape
    assert K == K2, f"Incompatible dimensions: A has {K} columns but B has {K2} rows"
    
    # Create output tensor
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    # Set block sizes for optimization
    BLOCK_SIZE_M = 1  # We process one batch element per program
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_L = 32  # Tile size for L dimension
    BLOCK_SIZE_K = 64  # Tile size for K dimension
    
    # Grid: one block for each (n, m) combination
    grid = (N, M)
    
    # Launch kernel
    matmul_3d_2d_kernel[grid](
        A, B, C,
        N, M, K, L,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L).
        """
        return triton_matmul_3d_2d(A, B)