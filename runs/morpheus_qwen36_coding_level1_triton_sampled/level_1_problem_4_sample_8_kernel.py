import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk, stride_b1,
    stride_cm, stride_c1,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * stride_am
    acc = 0.0
    
    # Each thread processes a chunk of K elements
    for k in range(0, K, BLOCK_K):
        offsets_k = k + tl.arange(0, BLOCK_K)
        mask_k = offsets_k < K
        
        a = tl.load(A_ptr + row_offset + offsets_k, mask=mask_k, other=0.0)
        b = tl.load(B_ptr + offsets_k, mask=mask_k, other=0.0)
        
        acc += tl.sum(a * b)
        
    # Reduce partial sums across all threads in the block
    acc = tl.sum(acc)
    tl.store(C_ptr + row_offset, acc)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    assert B.shape == (K, 1)
    
    C = torch.empty((M, 1), dtype=A.dtype, device=A.device)
    
    BLOCK_K = 1024
    num_warps = 32
    
    grid = (M,)
    
    matmul_kernel[grid](
        A, B, C,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_K=BLOCK_K,
        num_warps=num_warps
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)