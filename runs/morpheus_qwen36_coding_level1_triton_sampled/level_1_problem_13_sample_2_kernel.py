import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_m < N
    mask_n = offs_n < N
    mask_k = offs_k < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, N, BLOCK_K):
        # Load A block: A[offs_m, offs_k + k]
        A_ptrs = A_ptr + (offs_m[:, None] * N) + (offs_k[None, :] + k)
        A_block = tl.load(A_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        
        # Load B block using symmetry: B[k, n] = B[n, k]
        # Load B[n, k] which corresponds to row-major access for B
        B_ptrs = B_ptr + ((offs_n[:, None]) * N) + (offs_k[None, :] + k)
        B_block = tl.load(B_ptrs, mask=(mask_n[:, None] & mask_k[None, :]), other=0.0)
        
        # Transpose B block to shape (BLOCK_K, BLOCK_N) for dot product
        B_block_T = tl.trans(B_block)
        
        # Perform dot product with FP32 precision
        acc += tl.dot(A_block, B_block_T, allow_tf32=False)
        
    # Store result
    C_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(C_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    out = torch.empty_like(A)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    
    matmul_kernel[grid](
        A, B, out,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)