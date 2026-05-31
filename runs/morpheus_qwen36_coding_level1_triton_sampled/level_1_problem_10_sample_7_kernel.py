import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M, K, L,
    stride_A_N, stride_A_M, stride_A_K,
    stride_B_K, stride_B_L,
    stride_C_N, stride_C_M, stride_C_L,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Compute block indices from program ID
    pid = tl.program_id(0)
    num_m_blocks = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_l_blocks = (L + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L
    
    batch_idx = pid // (num_m_blocks * num_l_blocks)
    rest = pid % (num_m_blocks * num_l_blocks)
    row_block = rest // num_l_blocks
    col_block = rest % num_l_blocks
    
    # Generate offsets for rows, columns, and reduction dimension
    row_offsets = row_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_block * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for boundary handling
    mask_row = row_offsets < M
    mask_col = col_offsets < L
    mask_k = k_offsets < K
    
    # Initialize accumulator matrix
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Number of blocks in the K dimension
    num_k_blocks = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Loop over K blocks
    for k in range(num_k_blocks):
        # Compute base indices for the current K block
        k_base = k * BLOCK_SIZE_K
        
        # Compute absolute offsets for A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_row_offsets = row_offsets[:, None]
        a_k_offsets = k_base + k_offsets[None, :]
        
        # Compute absolute offsets for B: shape (BLOCK_SIZE_K, BLOCK_SIZE_L)
        b_k_offsets = k_base + k_offsets[:, None]
        b_col_offsets = col_offsets[None, :]
        
        # Compute masks for A and B blocks
        mask_a = (a_row_offsets < M) & (a_k_offsets < K)
        mask_b = (b_k_offsets < K) & (b_col_offsets < L)
        
        # Compute memory pointers for A and B blocks
        a_ptrs = A_ptr + (batch_idx * stride_A_N + 
                          a_row_offsets * stride_A_M + 
                          a_k_offsets * stride_A_K)
        b_ptrs = B_ptr + (b_k_offsets * stride_B_K + 
                          b_col_offsets * stride_B_L)
        
        # Load blocks with masking
        a_block = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b_block = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Perform dot product and accumulate
        acc += tl.dot(a_block, b_block)
        
    # Compute mask for C store
    c_mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < L)
    
    # Compute memory pointer for C block
    c_ptrs = C_ptr + (batch_idx * stride_C_N + 
                      row_offsets[:, None] * stride_C_M + 
                      col_offsets[None, :] * stride_C_L)
    
    # Store result
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton matmul kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    _, L = B.shape
    
    # Output tensor
    C = torch.empty((N, M, L), dtype=torch.float32, device='cuda')
    
    # Compute strides
    stride_A_N = M * K
    stride_A_M = K
    stride_A_K = 1
    
    stride_B_K = L
    stride_B_L = 1
    
    stride_C_N = M * L
    stride_C_M = L
    stride_C_L = 1
    
    # Block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_L = 64
    BLOCK_SIZE_K = 32
    
    # Grid configuration
    num_m_blocks = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_l_blocks = (L + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L
    grid = (N * num_m_blocks * num_l_blocks,)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        N, M, K, L,
        stride_A_N, stride_A_M, stride_A_K,
        stride_B_K, stride_B_L,
        stride_C_N, stride_C_M, stride_C_L,
        BLOCK_SIZE_M, BLOCK_SIZE_L, BLOCK_SIZE_K
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)