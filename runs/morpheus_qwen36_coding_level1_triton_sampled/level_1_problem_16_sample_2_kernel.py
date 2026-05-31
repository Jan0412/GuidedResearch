import torch
import torch.nn as nn
import triton
import triton.language as tl

# Tunable block sizes for GEMM
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for M and N dimensions
    m_off = pid_m * BLOCK_M + tl.arange(BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(BLOCK_N)
    k_off = tl.arange(BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        k_block_off = k + k_off
        
        # Load block from A: A is (K, M), we need (BLOCK_M, BLOCK_K)
        # Load shape (BLOCK_K, BLOCK_M) then transpose
        mask_a = (k_block_off[:, None] < K) & (m_off[None, :] < M)
        val_a = tl.load(A_ptr + k_block_off[:, None] * M + m_off[None, :], mask=mask_a, other=0.0)
        val_a = val_a.T
        
        # Load block from B: B is (K, N), shape (BLOCK_K, BLOCK_N)
        mask_b = (k_block_off[:, None] < K) & (n_off[None, :] < N)
        val_b = tl.load(B_ptr + k_block_off[:, None] * N + n_off[None, :], mask=mask_b, other=0.0)
        
        # Matrix multiply and accumulate
        acc += tl.dot(val_a, val_b)
        
    # Store result block to C
    mask_c = (m_off[:, None] < M) & (n_off[None, :] < N)
    tl.store(C_ptr + m_off[:, None] * N + n_off[None, :], acc, mask=mask_c)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton GEMM kernel.
    Computes C = A^T @ B where A is (K, M) and B is (K, N).
    """
    M = A.shape[1]
    K = A.shape[0]
    N = B.shape[1]
    
    assert K == B.shape[0], "Dimension mismatch in matrix multiplication"
    
    # Prepare output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Calculate grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), 1)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton GEMM kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A^T @ B using Triton kernel.
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)