import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Calculate program ID and block indices
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Create row and column indices for the current block
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Masks for boundary checks
    mask_row = row_idx < M
    mask_col = col_idx < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    # Since K is small (32), BLOCK_K=32 allows unrolling
    for k in range(0, K, BLOCK_K):
        # Load tile from A
        a_offsets = row_idx[:, None] * K + (k + tl.arange(0, BLOCK_K))[None, :]
        a_mask = (k + tl.arange(0, BLOCK_K)) < K
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load tile from B
        b_offsets = (k + tl.arange(0, BLOCK_K))[:, None] * N + col_idx[None, :]
        b_mask = (k + tl.arange(0, BLOCK_K)) < K
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Perform dot product and accumulate
        accumulator = tl.dot(a, b, accumulator)
        
    # Store result to C
    c_offsets = row_idx[:, None] * N + col_idx[None, :]
    tl.store(C_ptr + c_offsets, accumulator, mask=(mask_row[:, None] & mask_col[None, :]))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton kernel for matrix multiplication optimized for 
    the shape where one dimension is small (K=32) and output is large.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    _, N = B.shape
    
    assert K == B.shape[0], "Inner dimensions must match."
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Tunable block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32  # Matches the small K dimension exactly
    
    # Grid calculation
    num_blocks = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    grid = lambda meta: (num_blocks,)
    
    # Launch kernel
    matmul_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        M=M,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)