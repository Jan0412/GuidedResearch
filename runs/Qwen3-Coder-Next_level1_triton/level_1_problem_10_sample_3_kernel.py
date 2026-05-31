import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_3d_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M, K, L,
    stride_a0, stride_a1, stride_a2,
    stride_b0, stride_b1,
    stride_c0, stride_c1, stride_c2,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the batch index (n_idx) and the row index in the output matrix (m_idx)
    n_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    
    # Initialize pointer offsets for the output matrix C[n_idx, m_idx, :]
    c_offsets = tl.arange(0, BLOCK_SIZE_M) + m_idx * BLOCK_SIZE_M
    # We'll compute L output columns in chunks of BLOCK_SIZE_M (if L is not divisible, we'll mask)
    # But we'll use dynamic bounds to handle the exact L size
    c_ptrs = C_ptr + n_idx * stride_c0 + m_idx * BLOCK_SIZE_M * stride_c1 + tl.arange(0, BLOCK_SIZE_M) * stride_c2
    mask_c = c_offsets < L

    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Iterate over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        # Load A[n_idx, m_idx, k:k+BLOCK_SIZE_K]
        a_ptrs = A_ptr + n_idx * stride_a0 + m_idx * stride_a1 + k_offsets * stride_a2
        mask_a = k_offsets < K
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load B[k:k+BLOCK_SIZE_K, :]
        # Note: We need B[k, :] for all k in [k, k+BLOCK_SIZE_K)
        # We'll load B in chunks along K dimension, then broadcast to compute the matmul
        b_ptrs = B_ptr + k_offsets[:, None] * stride_b0 + tl.arange(0, BLOCK_SIZE_M)[None, :] * stride_b1
        mask_b_k = k_offsets[:, None] < K
        mask_b_m = tl.arange(0, BLOCK_SIZE_M)[None, :] < L
        mask_b = mask_b_k & mask_b_m
        # To handle the case where L might be larger than BLOCK_SIZE_M, we'll load only BLOCK_SIZE_M elements and mask accordingly
        # But we can use dynamic bounds to get only up to L columns
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Compute partial matmul: A[n_idx, m_idx, k:k+BLOCK_SIZE_K] * B[k:k+BLOCK_SIZE_K, :]
        # This is a dot product: (BLOCK_SIZE_K,) * (BLOCK_SIZE_K, BLOCK_SIZE_M) -> (BLOCK_SIZE_M,)
        # But since B is (K, L), we want only the first BLOCK_SIZE_M columns (masked)
        # Actually, we want to compute the dot product with all L columns, so we need to handle L properly
        
        # Let's restructure: We'll compute only BLOCK_SIZE_M columns at a time and mask
        # Actually, for simplicity and correctness, we'll compute the full L dimension but only process BLOCK_SIZE_M columns at a time
        # So let's adjust: The outer loop is over n_idx and m_idx, and for each (n_idx, m_idx), we compute L elements
        # But since we're in the kernel, we want to use BLOCK_SIZE_M for the M dimension, not L
        # Wait, let me correct: We have C of shape (N, M, L), and we're processing in blocks over M and L
        # Actually, let me fix the kernel: We want to iterate over blocks of M and L, not M and M
        
        # Correction: Let's change the kernel to iterate over M and L blocks
        # We'll fix the kernel below
        pass
    
    # Store the result
    tl.store(c_ptrs + tl.arange(0, BLOCK_SIZE_M), acc, mask=mask_c)

# Let me rewrite the kernel with better tiling strategy
# We'll tile over M and L dimensions, and iterate over K
@triton.jit
def matmul_3d_kernel_v2(
    A_ptr, B_ptr, C_ptr,
    N, M, K, L,
    stride_a0, stride_a1, stride_a2,
    stride_b0, stride_b1,
    stride_c0, stride_c1, stride_c2,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # n_idx: batch index
    # m_idx: row index in the M dimension of output
    # l_idx: column index in the L dimension of output
    n_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    l_idx = tl.program_id(2)

    # Compute the starting offset for this block in the M and L dimensions
    m_offsets = m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    l_offsets = l_idx * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    
    # Create meshgrid for the output block: (BLOCK_SIZE_M, BLOCK_SIZE_L)
    m_grid, l_grid = tl.meshgrid(m_offsets, l_offsets)
    m_grid = m_grid.T  # Transpose to get (BLOCK_SIZE_M, BLOCK_SIZE_L) shape
    l_grid = l_grid.T

    # Mask for valid indices
    mask_m = m_grid < M
    mask_l = l_grid < L
    mask = mask_m & mask_l

    # Initialize accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)

    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < K
        
        # Load A[n_idx, m_grid, k_offsets]
        # A shape: (N, M, K), so we need to compute indices for A[n_idx, m, k]
        # m_grid is (BLOCK_SIZE_M, BLOCK_SIZE_L), so we need to broadcast k_offsets
        # A indices: [n_idx, m_grid, k_offsets]
        a_indices = n_idx * stride_a0 + m_grid * stride_a1 + k_offsets[None, None, :] * stride_a2
        # But we need to reshape for proper broadcasting
        # Let's compute A values for each (m, l) pair and k
        # Since A is (N, M, K), and we want A[n_idx, m, k] for m in m_offsets and k in k_offsets
        # We'll compute A values separately for each m in m_offsets
        
        # Actually, let's simplify: we'll compute the dot product for each (m, l) pair
        # For a fixed (m, l), C[n_idx, m, l] = sum_k A[n_idx, m, k] * B[k, l]
        # So we can compute this directly
        
        # Let's change to a simpler approach: for each block of (m, l), compute the dot product over k
        # We'll compute C[n_idx, m, l] for m in [m_idx*BLOCK_SIZE_M, (m_idx+1)*BLOCK_SIZE_M) and l in [l_idx*BLOCK_SIZE_L, (l_idx+1)*BLOCK_SIZE_L)
        
        # Load A[n_idx, m_offsets, k:k+BLOCK_SIZE_K]
        a_m_k = tl.load(
            A_ptr + n_idx * stride_a0 + m_grid * stride_a1 + k_offsets[None, None, :] * stride_a2,
            mask=mask_m[:, :, None] & mask_k[None, None, :],
            other=0.0
        )
        
        # Load B[k:k+BLOCK_SIZE_K, l_offsets]
        b_k_l = tl.load(
            B_ptr + k_offsets[:, None, None] * stride_b0 + l_grid[None, :, :] * stride_b1,
            mask=mask_k[:, None, None] & mask_l[None, :, :],
            other=0.0
        )
        
        # Compute the outer product and accumulate
        # a_m_k is (BLOCK_SIZE_M, BLOCK_SIZE_L, BLOCK_SIZE_K)
        # b_k_l is (BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_L)
        # We want to compute sum over k: a_m_k[m, l, k] * b_k_l[k, m, l]
        # Actually, let's reshape and multiply
        a_m_k = a_m_k * b_k_l
        acc += tl.sum(a_m_k, axis=2)  # Sum over K dimension
    
    # Store the result
    c_vals = acc.to(tl.float32)
    tl.store(
        C_ptr + n_idx * stride_c0 + m_grid * stride_c1 + l_grid * stride_c2,
        c_vals,
        mask=mask
    )

# Let me simplify further: since this is a 3D tensor-matrix multiplication, we can treat it as N separate (M, K) x (K, L) matrix multiplications
# So we can use a standard 2D matmul kernel and iterate over N
@triton.jit
def matmul_3d_kernel_simple(
    A_ptr, B_ptr, C_ptr,
    N, M, K, L,
    stride_a0, stride_a1, stride_a2,
    stride_b0, stride_b1,
    stride_c0, stride_c1, stride_c2,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # This is the batch block size
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    # We'll iterate over batch dimension (N) in blocks
    n_block = tl.program_id(0)
    m_block = tl.program_id(1)
    l_block = tl.program_id(2)
    
    # Compute offsets for the current batch block, M block, and L block
    n_offsets = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_offsets = m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    l_offsets = l_block * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    
    # Create meshgrid for M and L dimensions
    m_grid, l_grid = tl.meshgrid(m_offsets, l_offsets)
    m_grid = m_grid.T
    l_grid = l_grid.T
    
    # Mask for valid indices
    mask_m = m_grid < M
    mask_l = l_grid < L
    mask_n = n_offsets < N
    mask = mask_m & mask_l & mask_n[:, None, None]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < K
        
        # Load A[n_offsets, m_grid, k_offsets]
        # A shape: (N, M, K)
        # We need A[n, m, k] for n in n_offsets, m in m_offsets, k in k_offsets
        # Compute indices: A_ptr + n*stride_a0 + m*stride_a1 + k*stride_a2
        a_indices = n_offsets[:, None, None, None] * stride_a0 + \
                   m_grid[None, :, :, :] * stride_a1 + \
                   k_offsets[None, None, None, :] * stride_a2
        a_vals = tl.load(A_ptr + a_indices, mask=mask_m[None, :, :, :] & mask_k[None, None, None, :], other=0.0)
        
        # Load B[k_offsets, l_grid]
        # B shape: (K, L)
        b_indices = k_offsets[:, None, None, None] * stride_b0 + \
                   l_grid[None, :, :, :] * stride_b1
        b_vals = tl.load(B_ptr + b_indices, mask=mask_k[:, None, None, None] & mask_l[None, :, :, :], other=0.0)
        
        # Compute accumulation: sum over k of A[n, m, k] * B[k, l]
        # a_vals shape: (BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_L, BLOCK_SIZE_K)
        # b_vals shape: (BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_L)
        # We want to sum over the K dimension (axis 3 for a_vals, axis 0 for b_vals)
        acc += tl.sum(a_vals * b_vals, axis=3)
    
    # Store result
    c_indices = n_offsets[:, None, None] * stride_c0 + \
               m_grid[None, :, :] * stride_c1 + \
               l_grid[None, :, :] * stride_c2
    tl.store(C_ptr + c_indices, acc, mask=mask)

# Let's go back to the original problem: we want to compute C[n, m, l] = sum_k A[n, m, k] * B[k, l]
# This is equivalent to doing N separate (M, K) x (K, L) matrix multiplications
# So let's use a standard 2D matmul kernel and loop over N
@triton.jit
def matmul_kernel_2d(
    A_ptr, B_ptr, C_ptr,
    M, K, L,
    stride_a0, stride_a1,
    stride_b0, stride_b1,
    stride_c0, stride_c1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    # m_block: row block index
    # l_block: column block index
    m_block = tl.program_id(0)
    l_block = tl.program_id(1)
    
    # Compute offsets for M and L dimensions
    m_offsets = m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    l_offsets = l_block * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    
    # Create meshgrid
    m_grid, l_grid = tl.meshgrid(m_offsets, l_offsets)
    m_grid = m_grid.T
    l_grid = l_grid.T
    
    # Mask for valid indices
    mask_m = m_grid < M
    mask_l = l_grid < L
    mask = mask_m & mask_l
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < K
        
        # Load A[m_grid, k_offsets]
        a_vals = tl.load(
            A_ptr + m_grid * stride_a0 + k_offsets[None, None, :] * stride_a1,
            mask=mask_m[:, :, None] & mask_k[None, None, :],
            other=0.0
        )
        
        # Load B[k_offsets, l_grid]
        b_vals = tl.load(
            B_ptr + k_offsets[:, None, None] * stride_b0 + l_grid[None, :, :] * stride_b1,
            mask=mask_k[:, None, None] & mask_l[None, :, :],
            other=0.0
        )
        
        # Sum over K dimension
        acc += tl.sum(a_vals * b_vals, axis=2)
    
    # Store result
    tl.store(
        C_ptr + m_grid * stride_c0 + l_grid * stride_c1,
        acc,
        mask=mask
    )

def triton_matmul_3d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 3D tensor-matrix multiplication using Triton kernels.
    
    Args:
        A (torch.Tensor): Input 3D tensor of shape (N, M, K).
        B (torch.Tensor): Input matrix of shape (K, L).
    
    Returns:
        torch.Tensor: Output tensor of shape (N, M, L).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    K_b, L = B.shape
    assert K == K_b, f"Inner dimensions must match: A.shape={A.shape}, B.shape={B.shape}"
    
    # Prepare output tensor
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_L = 64
    BLOCK_SIZE_K = 32
    
    # Compute grid dimensions
    grid_M = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_L = (L + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L
    grid_N = N  # We'll process each batch separately
    
    # Launch kernel for each batch
    for n in range(N):
        # Compute pointers for the current batch
        A_ptr_n = A.data_ptr() + n * A.stride(0) * A.element_size()
        C_ptr_n = C.data_ptr() + n * C.stride(0) * C.element_size()
        
        # Compute strides for the current batch
        # A[n, :, :] has shape (M, K), strides are A.stride(1), A.stride(2)
        # B has shape (K, L), strides are B.stride(0), B.stride(1)
        # C[n, :, :] has shape (M, L), strides are C.stride(1), C.stride(2)
        
        matmul_kernel_2d[grid_M, grid_L](
            A_ptr_n, B.data_ptr(), C_ptr_n,
            M, K, L,
            A.stride(1), A.stride(2),
            B.stride(0), B.stride(1),
            C.stride(1), C.stride(2),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_L=BLOCK_SIZE_L,
        )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized version of the Model that uses custom Triton kernels for 3D tensor-matrix multiplication.
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
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return triton_matmul_3d(A, B)