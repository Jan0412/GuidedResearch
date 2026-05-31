import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    m, k, n,
    stride_ab, stride_ak,
    stride_bb, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    A_batch_ptr = A_ptr + pid * stride_ab
    B_batch_ptr = B_ptr + pid * stride_bb
    C_batch_ptr = C_ptr + pid * stride_cb
    
    num_blocks_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_k = (k + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    num_blocks_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    for off_m in range(num_blocks_m):
        for off_n in range(num_blocks_n):
            offs_m = off_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = off_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            
            c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            
            for off_k in range(num_blocks_k):
                offs_k = off_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                
                a_ptrs = A_batch_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ak
                b_ptrs = B_batch_ptr + offs_k[:, None] * stride_bn + offs_n[None, :] * stride_cn
                
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                
                c = tl.dot(a, b, c)
                
            c_ptrs = C_batch_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, c)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    C = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)
    
    stride_ab = A.stride(0)
    stride_ak = A.stride(2)
    stride_bb = B.stride(0)
    stride_bn = B.stride(2)
    stride_cb = C.stride(0)
    stride_cm = C.stride(1)
    stride_cn = C.stride(2)
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_N = 128
    
    grid = (batch_size,)
    bmm_kernel[grid](
        A, B, C,
        m, k, n,
        stride_ab, stride_ak,
        stride_bb, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)