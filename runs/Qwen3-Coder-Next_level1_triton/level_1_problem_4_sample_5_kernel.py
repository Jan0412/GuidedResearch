import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matvec_kernel(
    A_ptr,  # Matrix A of shape (M, K)
    B_ptr,  # Vector B of shape (K,)
    C_ptr,  # Output vector of shape (M,)
    M, K,
    stride_am, stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID corresponds to the row in matrix A
    pid = tl.program_id(0)
    
    # Offset for the current row
    row_start = pid * BLOCK_SIZE_M
    
    # Load a block of rows from A (up to BLOCK_SIZE_M rows)
    # We'll process in chunks of BLOCK_SIZE_K along the K dimension
    for m_offset in range(BLOCK_SIZE_M):
        row_idx = row_start + m_offset
        if row_idx >= M:
            break
            
        # Accumulator for this row
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Process K dimension in blocks
        for k_start in range(0, K, BLOCK_SIZE_K):
            k_end = tl.minimum(k_start + BLOCK_SIZE_K, K)
            k_range = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_range < K
            
            # Load the row slice from A: shape (BLOCK_SIZE_K,)
            a_ptrs = A_ptr + row_idx * stride_am + k_range * stride_ak
            a_vals = tl.load(a_ptrs, mask=k_mask, other=0.0)
            
            # Load the corresponding slice from B: shape (BLOCK_SIZE_K,)
            b_ptrs = B_ptr + k_range * stride_bk
            b_vals = tl.load(b_ptrs, mask=k_mask, other=0.0)
            
            # Accumulate dot product
            acc += tl.sum(a_vals * b_vals)
        
        # Store the result
        c_ptr = C_ptr + row_idx * stride_cm
        tl.store(c_ptr, acc)


def triton_matvec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector multiplication using Triton kernel.
    A: (M, K) matrix
    B: (K,) vector
    Returns: (M,) vector
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    assert B.shape == (K,), f"Expected B shape ({K},), got {B.shape}"
    
    # Prepare output tensor
    C = torch.empty(M, dtype=A.dtype, device=A.device)
    
    # Set up block sizes (tunable parameters)
    BLOCK_SIZE_M = 32  # Rows processed per block
    BLOCK_SIZE_K = 256  # K dimension block size
    
    # Grid: one block per BLOCK_SIZE_M rows
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,)
    
    # Launch kernel
    matvec_kernel[grid](
        A, B, C,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0),
        C.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized Triton kernel.
        """
        return triton_matvec(A, B)