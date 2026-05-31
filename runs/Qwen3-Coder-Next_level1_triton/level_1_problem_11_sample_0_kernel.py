import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_4d_kernel(
    A_ptr,  # Pointer to 4D tensor A with shape (b, i, j, l)
    B_ptr,  # Pointer to matrix B with shape (l, k)
    C_ptr,  # Pointer to output 4D tensor C with shape (b, i, j, k)
    B_dim,  # Leading dimension of B (number of columns in B, which is k)
    B_ld,   # Leading dimension of B in memory (stride between rows of B)
    B_stride,  # Stride between rows of B in memory
    total_batches,  # Total number of batches (b)
    total_rows,     # Total number of i*j blocks (i*j)
    k_size,         # Output dimension k
    l_size,         # Inner dimension l
    BLOCK_M: tl.constexpr,  # Block size for output rows
    BLOCK_N: tl.constexpr,  # Block size for output columns
    BLOCK_K: tl.constexpr,  # Block size for reduction dimension
):
    # Compute batch and block indices
    batch_id = tl.program_id(2)
    block_m_id = tl.program_id(0)
    block_n_id = tl.program_id(1)

    # Offsets for batch
    batch_offset = batch_id * total_rows * k_size * BLOCK_M

    # Compute row offset (m) for output matrix
    m_offsets = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    # Compute column offset (n) for output matrix
    n_offsets = block_n_id * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create full offsets for output tensor (b, i, j, k)
    # We'll handle flattened i*j as rows in the matrix multiplication view
    row_idx = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = block_n_id * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator for C matrix block
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the reduction dimension (l)
    for k_block in range(0, l_size, BLOCK_K):
        k_offsets = k_block + tl.arange(0, BLOCK_K)
        
        # Load A block: shape (BLOCK_M, BLOCK_K)
        # A is indexed as A[batch_id, row_idx // j, row_idx % j, k_offsets]
        # We flatten the i*j dimension to treat it as a 2D matmul
        a_offsets = batch_offset + (row_idx[:, None] // (i_size // total_batches) * j_size * l_size) + \
                    (row_idx[:, None] % (j_size // total_batches) * l_size) + k_offsets[None, :]
        
        # Actually, let's simplify by computing based on actual tensor dimensions
        # We'll handle this with proper indexing
        a_m = row_idx // (j_size // (total_batches // b))
        a_n = row_idx % (j_size // (total_batches // b))
        
        # Let's use a different approach with explicit indexing
        # For simplicity, we'll compute row/col indices in the original 4D structure
        # and use them to access A properly
        
        # Re-calculate indices with respect to original dimensions
        row_m = batch_id * (i_size * j_size) + (row_idx // j_size) * j_size + (row_idx % j_size)
        a_offsets = row_m[:, None] * l_size + k_offsets[None, :]
        
        # Load B block: shape (BLOCK_K, BLOCK_N)
        b_offsets = k_offsets[:, None] * B_ld + col_idx[None, :]
        
        # Load A and B
        a = tl.load(A_ptr + a_offsets, mask=(row_idx[:, None] < total_rows) & (k_offsets[None, :] < l_size), other=0.0)
        b = tl.load(B_ptr + b_offsets, mask=(k_offsets[:, None] < l_size) & (col_idx[None, :] < k_size), other=0.0)
        
        # Accumulate product
        acc += tl.dot(a, b)

    # Store result
    c_offsets = row_idx[:, None] * k_size + col_idx[None, :]
    c = acc.to(tl.float32)
    
    # Store with mask to handle out-of-bounds elements
    tl.store(C_ptr + c_offsets, c, mask=(row_idx[:, None] < total_rows) & (col_idx[None, :] < k_size))


# Global variables to store dimensions (will be set during kernel launch)
i_size = 256
j_size = 512
b_size = 8


def triton_matmul_4d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 4D tensor-matrix multiplication: C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    
    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)
        
    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    b, i, j, l = A.shape
    l2, k = B.shape
    
    # Create output tensor
    C = torch.empty((b, i, j, k), dtype=A.dtype, device=A.device)
    
    # Define block sizes for kernel
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Calculate grid dimensions
    # For M dimension: ceil(i*j / BLOCK_M)
    # For N dimension: ceil(k / BLOCK_N)
    # For batch dimension: b
    grid_m = (i * j + BLOCK_M - 1) // BLOCK_M
    grid_n = (k + BLOCK_N - 1) // BLOCK_N
    
    # Launch kernel
    matmul_4d_kernel[(grid_m, grid_n, b)](
        A, B, C,
        k,  # B_dim
        k,  # B_ld (leading dimension)
        k,  # B_stride
        b * i * j,  # total_batches (treat all batches as separate)
        i * j,      # total_rows
        k,          # k_size
        l,          # l_size
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return C


# Global variables for kernel dimensions
matmul_4d_kernel.global_keys = ['i_size', 'j_size', 'b_size']


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 4D tensor-matrix multiplication.
    Performs: C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
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
        global i_size, j_size, b_size
        
        # Set global variables for kernel dimensions
        b, i, j, l = A.shape
        i_size = i
        j_size = j
        b_size = b
        
        return triton_matmul_4d(A, B)