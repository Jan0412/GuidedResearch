import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_4d_kernel(
    A_ptr, B_ptr, C_ptr,
    B_dim, I_dim, J_dim, L_dim, K_dim,
    stride_ab, stride_ai, stride_aj, stride_al,
    stride_bl, stride_bk,
    stride_cb, stride_ci, stride_cj, stride_ck,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Batch and row indices
    batch_idx = tl.program_id(0)
    i_idx = tl.program_id(1)
    j_idx = tl.program_id(2)
    
    # Calculate the start offsets for this block
    # Output block: M = BLOCK_SIZE_M (columns of C), N = BLOCK_SIZE_N (rows of C)
    # But for this einsum, we're computing for fixed batch, i, j -> k dimension
    # So we parallelize over k dimension (BLOCK_SIZE_N) and use BLOCK_SIZE_M for unrolling
    
    # Actually, let's restructure: parallelize over batch*i*j and compute k dimension
    # But given the dimensions, better to parallelize over (batch, i, j) and compute k
    
    # For this implementation, we'll compute one (batch, i, j) block at a time across k
    # Each program handles one (batch, i, j) and computes a tile of k
    
    # However, with j=512 and k=768, let's use a more efficient parallelization:
    # Program 0: batch index
    # Program 1: i index  
    # Program 2: tile index over j*k
    
    # Let's try a different approach: flatten (i, j) into one dimension
    # So we have (b, i*j, l) @ (l, k) -> (b, i*j, k)
    
    # But since the kernel signature above assumes 3D grid, let's use:
    # batch_idx, i_idx, and tile over j*k combined
    
    # Actually, for optimal performance, let's parallelize over (batch, i, j) and compute k
    # Since j=512 and k=768, we can compute k in blocks
    
    # Get the starting position for this block
    k_block_start = tl.program_id(2) * BLOCK_SIZE_N
    
    # Compute the linear index for the (i, j) pair
    ij_idx = i_idx * J_dim + j_idx
    
    # Pointers to the input tensors
    # A is (b, i, j, l), so for given batch and (i,j), we need A[batch, i, j, :]
    a_ptr = A_ptr + batch_idx * stride_ab + i_idx * stride_ai + j_idx * stride_aj
    b_ptr = B_ptr
    
    # Pointer to the output
    c_ptr = C_ptr + batch_idx * stride_cb + i_idx * stride_ci + j_idx * stride_cj + k_block_start * stride_ck
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_N,), tl.float32)
    
    # Loop over K dimension (l dimension of A)
    for k_idx in range(0, L_dim, BLOCK_SIZE_K):
        # Load a tile of A: shape (BLOCK_SIZE_K,)
        a_offsets = k_idx + tl.arange(0, BLOCK_SIZE_K)
        a_mask = a_offsets < L_dim
        a = tl.load(a_ptr + a_offsets * stride_al, mask=a_mask, other=0.0)
        
        # Load a tile of B: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_offsets_k = k_idx + tl.arange(0, BLOCK_SIZE_K)
        b_offsets_n = k_block_start + tl.arange(0, BLOCK_SIZE_N)
        b_mask_k = b_offsets_k < L_dim
        b_mask_n = b_offsets_n < K_dim
        b_mask = b_mask_k[:, None] & b_mask_n[None, :]
        
        b = tl.load(b_ptr + b_offsets_k[:, None] * stride_bl + b_offsets_n[None, :] * stride_bk, mask=b_mask, other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(a[:, None] * b, axis=0)
    
    # Store the result
    c_offsets = tl.arange(0, BLOCK_SIZE_N)
    c_mask = c_offsets < (K_dim - k_block_start)
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)


def triton_matmul_4d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 4D tensor-matrix multiplication using Triton kernel:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    b, i, j, l = A.shape
    l2, k = B.shape
    assert l == l2, f"Dimension mismatch: A has l={l}, B has l={l2}"
    
    # Create output tensor
    C = torch.empty((b, i, j, k), dtype=A.dtype, device=A.device)
    
    # Define kernel parameters
    # Block sizes for efficient computation
    BLOCK_SIZE_M = 1  # Not used in this implementation
    BLOCK_SIZE_N = 64  # Tile size for K dimension
    BLOCK_SIZE_K = 32  # Tile size for L dimension
    
    # Grid: (batch, i, ceil(j*k / BLOCK_SIZE_N)) - but better to parallelize differently
    # Actually, let's parallelize over (batch, i, j) and compute k in blocks
    grid = (b, i, (k + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    # Launch kernel
    matmul_4d_kernel[grid](
        A, B, C,
        b, i, j, l, k,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 4D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return triton_matmul_4d(A, B)