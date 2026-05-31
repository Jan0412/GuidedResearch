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
    
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    
    A_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        mask_k = (k + offs_k) < K
        A = tl.load(A_ptrs, mask=mask_am[:, None] & mask_k[None, :], other=0.0)
        B = tl.load(B_ptrs, mask=mask_bn[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(A, B, transpose_b=True)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
        
    C_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=mask_am[:, None] & mask_bn[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    N, _ = B.shape
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    num_warps = 4
    num_stages = 2
    
    stride_am, stride_ak = A.stride()
    stride_bn, stride_bk = B.stride()
    stride_cm, stride_cn = C.stride()
    
    grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"], 
                         (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"], 1)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(N, K)
    return [A, B]

def get_init_inputs():
    return []