import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = m_offsets < N
    mask_n = n_offsets < M
    
    # Load A values for the current block of rows
    A_vals = tl.load(A_ptr + m_offsets, mask=mask_m, other=0.0)
    
    # Compute pointers for B
    B_ptrs = B_ptr + m_offsets[:, None] * M + n_offsets[None, :]
    B_vals = tl.load(B_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Perform element-wise multiplication: C[i, j] = A[i] * B[i, j]
    C_vals = A_vals[:, None] * B_vals
    
    # Store result
    tl.store(C_ptr + m_offsets[:, None] * M + n_offsets[None, :], C_vals, mask=mask_m[:, None] & mask_n[None, :])


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    M = B.shape[1]
    
    C = torch.empty_like(B)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    grid = ((N + BLOCK_M - 1) // BLOCK_M, (M + BLOCK_N - 1) // BLOCK_N, 1)
    
    diag_matmul_kernel[grid](A, B, C, N, M, BLOCK_M, BLOCK_N)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)