import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_am < M
    mask_n = offs_bn < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # Load A block: shape (BLOCK_K, BLOCK_M) -> transpose to (BLOCK_M, BLOCK_K)
        A_block = tl.load(
            A_ptr + (k + offs_k[:, None]) * stride_ak + (pid_m * BLOCK_M + offs_am[None, :]) * stride_am,
            mask=(k + offs_k[:, None] < K) & (pid_m * BLOCK_M + offs_am[None, :] < M),
            other=0.0
        )
        A_block = tl.trans(A_block)
        
        # Load B block: shape (BLOCK_N, BLOCK_K) -> transpose to (BLOCK_K, BLOCK_N)
        B_block = tl.load(
            B_ptr + (pid_n * BLOCK_N + offs_bn[:, None]) * stride_bn + (k + offs_k[None, :]) * stride_bk,
            mask=(pid_n * BLOCK_N + offs_bn[:, None] < N) & (k + offs_k[None, :] < K),
            other=0.0
        )
        B_block = tl.trans(B_block)
        
        acc = tl.dot(A_block, B_block, acc=acc, out_dtype=tl.float32)
        
    # Store result
    tl.store(
        C_ptr + (pid_m * BLOCK_M + offs_am[:, None]) * stride_cm + (pid_n * BLOCK_N + offs_bn[None, :]) * stride_cn,
        acc,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    M = A.shape[1]
    N = B.shape[0]
    K = A.shape[0]
    
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=2
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)