import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def upper_triangular_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_a_row, stride_a_col,
    stride_b_row, stride_b_col,
    stride_c_row, stride_c_col,
    BLOCK_M, BLOCK_N, BLOCK_K
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_i = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_i = offs_i < N
    mask_j = offs_j < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, N, BLOCK_K):
        # Load A tile with mask i <= k+k
        mask_a = (offs_i[:, None] <= k + offs_k[None, :])
        mask_a &= mask_i[:, None]
        mask_a &= (k + offs_k)[None, :] < N
        a = tl.load(
            A_ptr + offs_i[:, None] * stride_a_row + (k + offs_k)[None, :] * stride_a_col,
            mask=mask_a, other=0.0
        )
        
        # Load B tile with mask k+k <= j
        mask_b = (k + offs_k)[:, None] <= offs_j[None, :]
        mask_b &= (k + offs_k)[:, None] < N
        mask_b &= mask_j[None, :]
        b = tl.load(
            B_ptr + (k + offs_k)[:, None] * stride_b_row + offs_j[None, :] * stride_b_col,
            mask=mask_b, other=0.0
        )
        
        acc += tl.dot(a, b)
        
    # Store result with mask i <= j
    mask_c = offs_i[:, None] <= offs_j[None, :]
    mask_c &= mask_i[:, None]
    mask_c &= mask_j[None, :]
    
    tl.store(
        C_ptr + offs_i[:, None] * stride_c_row + offs_j[None, :] * stride_c_col,
        acc, mask=mask_c
    )


def triton_upper_triangular_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    N = A.shape[0]
    C = torch.empty((N, N), dtype=A.dtype, device=A.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    upper_triangular_matmul_kernel[grid](
        A, B, C, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A, B):
        return triton_upper_triangular_matmul(A, B)