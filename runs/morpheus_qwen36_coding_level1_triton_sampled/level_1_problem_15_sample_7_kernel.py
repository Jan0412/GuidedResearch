import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tri_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Only compute lower triangular part
    if pid_n > pid_m:
        return
        
    acc = 0.0
    # k ranges from 0 to pid_n
    for k in range(0, pid_n + 1, BLOCK_SIZE):
        offsets_k = k + tl.arange(0, BLOCK_SIZE)
        mask_k = offsets_k <= pid_n
        
        a = tl.load(A_ptr + pid_m * stride_am + offsets_k * stride_ak, mask=mask_k, other=0.0)
        b = tl.load(B_ptr + offsets_k * stride_bk + pid_n * stride_bn, mask=mask_k, other=0.0)
        
        acc += tl.sum(a * b, axis=0)
        
    tl.store(C_ptr + pid_m * stride_cm + pid_n * stride_cn, acc)


def triton_tri_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    M = A.shape[0]
    out = torch.empty_like(A)
    
    BLOCK_SIZE = 32
    grid = (M, M)
    
    tri_matmul_kernel[grid](
        A, B, out,
        M,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        # Since A and B are lower triangular, their product is automatically lower triangular.
        # We skip torch.tril and compute only the lower triangular part efficiently.
        return triton_tri_matmul(A, B)