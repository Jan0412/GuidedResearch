import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matvec_kernel(
    A_ptr,           # Pointer to matrix A
    B_ptr,           # Pointer to vector B
    Out_ptr,         # Pointer to output vector
    M,               # Number of rows in A
    K,               # Number of columns in A / length of B
    stride_am,       # Stride between rows of A
    stride_ak,       # Stride between columns of A
    stride_bk,       # Stride between elements of B
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID for the block of rows we are processing
    pid = tl.program_id(0)
    
    # Row offsets for this program
    rm = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Column offsets for the tiling of K
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Pointers to the start of the blocks
    # A is (M, K), B is (K, 1)
    # We load a block of rows from A and a block of elements from B
    a_ptr = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr = B_ptr + (rk[None, :] * stride_bk)
    
    # Loop over the K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Masking for the K dimension to handle cases where K is not a multiple of BLOCK_SIZE_K
        k_mask = rk < (K - k)
        # Masking for the M dimension to handle cases where M is not a multiple of BLOCK_SIZE_M
        m_mask = rm < M
        
        # Load A block (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = tl.load(a_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # Load B block (1, BLOCK_SIZE_K)
        b = tl.load(b_ptr, mask=k_mask[None, :], other=0.0)
        
        # Compute dot product for each row in the block
        # a * b performs broadcasting
        accumulator += tl.sum(a * b, axis=1)
        
        # Advance pointers
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk
        
    # Store the result
    out_offsets = rm
    tl.store(Out_ptr + out_offsets, accumulator, mask=m_mask)

def triton_matvec(A: torch.Tensor, B: torch.Tensor):
    # Ensure inputs are on CUDA and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    # B is (K, 1), we treat it as (K,)
    
    # Output tensor (M, 1)
    out = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    # Tuning parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_K = 1024
    
    # Grid: one program per block of M rows
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,)
    
    matvec_kernel[grid](
        A, B, out,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication (C = A * B) using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using a custom Triton kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matvec(A, B)