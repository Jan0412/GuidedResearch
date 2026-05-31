import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_3d_kernel(
    A_ptr,  # Pointer to input tensor A of shape (N, M, K)
    B_ptr,  # Pointer to input matrix B of shape (K, L)
    C_ptr,  # Pointer to output tensor C of shape (N, M, L)
    N, M, K, L,
    stride_a0, stride_a1, stride_a2,  # Strides for A: (M*K, K, 1)
    stride_b0, stride_b1,            # Strides for B: (L, 1)
    stride_c0, stride_c1, stride_c2, # Strides for C: (M*L, L, 1)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    # Program IDs: 
    # pid_n = program_id(0) -> batch index n in [0, N)
    # pid_m = program_id(1) -> row index m in [0, M)
    # pid_l_block = program_id(2) -> block index for L dimension
    
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_l_block = tl.program_id(2)
    
    # Offset for this (n, m) pair
    a_offset = pid_n * stride_a0 + pid_m * stride_a1
    
    # Compute which part of L dimension this program handles
    l_start = pid_l_block * BLOCK_SIZE_L
    l_offsets = l_start + tl.arange(0, BLOCK_SIZE_L)
    l_mask = l_offsets < L
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_L,), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        # Load A slice: shape (BLOCK_SIZE_K,)
        a_ptrs = A_ptr + a_offset + k_offsets * stride_a2
        a_ptrs = tl.max_contiguous(tl.multiple_of(a_ptrs, 16), BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        
        # Load B slice: shape (BLOCK_SIZE_K, BLOCK_SIZE_L)
        b_ptrs = B_ptr + k_offsets[:, None] * stride_b0 + l_offsets[None, :] * stride_b1
        b = tl.load(b_ptrs, mask=k_mask[:, None] & l_mask[None, :], other=0.0)
        
        # Accumulate: a (BLOCK_SIZE_K,) @ b (BLOCK_SIZE_K, BLOCK_SIZE_L)
        acc += tl.sum(a[:, None] * b, axis=0)
    
    # Store result
    c_ptrs = C_ptr + pid_n * stride_c0 + pid_m * stride_c1 + l_offsets * stride_c2
    c_ptrs = tl.max_contiguous(tl.multiple_of(c_ptrs, 16), BLOCK_SIZE_L)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=l_mask)


def triton_matmul_3d(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized 3D tensor-matrix multiplication using Triton.
    
    Args:
        A (torch.Tensor): Input 3D tensor of shape (N, M, K)
        B (torch.Tensor): Input matrix of shape (K, L)
    
    Returns:
        torch.Tensor: Output tensor of shape (N, M, L)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    K_b, L = B.shape
    assert K == K_b, f"Incompatible dimensions: A has K={K}, B has K={K_b}"
    
    # Create output tensor
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_L = 32
    
    # Calculate grid dimensions
    grid_n = N
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_l = (L + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L
    
    grid = (grid_n, grid_m, grid_l)
    
    # Launch kernel
    matmul_3d_kernel[grid](
        A, B, C,
        N, M, K, L,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for 3D tensor-matrix multiplication.
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
        return triton_matmul_3d(A, B)