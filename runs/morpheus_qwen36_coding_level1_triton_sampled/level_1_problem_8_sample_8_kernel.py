import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_ak = tl.arange(0, BLOCK_SIZE_K)
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_bk = tl.arange(0, BLOCK_SIZE_K)

    mask_am = offsets_am < M
    mask_ak = offsets_ak < K
    mask_bn = offsets_bn < N
    mask_bk = offsets_bk < K

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_offsets = offsets_am[:, None] * stride_am + offsets_ak[None, :] * stride_ak
        a_mask = mask_am[:, None] & mask_ak[None, :]
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)

        b_offsets = offsets_bk[:, None] * stride_bk + offsets_bn[None, :] * stride_bn
        b_mask = mask_bk[:, None] & mask_bn[None, :]
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

    offsets_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_offsets = offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn
    c_mask = mask_am[:, None] & mask_bn[None, :]
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Inputs must be FP32."
    
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Incompatible dimensions for matmul."
    
    C = torch.empty((M, N), dtype=torch.float32, device='cuda')
    
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128), 1)
    
    matmul_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        M=M,
        N=N,
        K=K,
        stride_am=stride_am,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        stride_bn=stride_bn,
        stride_cm=stride_cm,
        stride_cn=stride_cn,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)