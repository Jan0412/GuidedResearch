import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_lower_tri_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = row_offsets < M
    mask_n = col_offsets < N
    mask_block = mask_m[:, None] & mask_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        
        a_ptrs = A + row_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = B + k_offsets[:, None] * stride_bk + col_offsets[None, :] * stride_bn

        mask_a = mask_m[:, None] & (k_offsets[None, :] < K)
        mask_b = (k_offsets[:, None] < K) & mask_n[None, :]

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b, out_dtype=tl.float32, num_stages=3)

    lower_tri_mask = row_offsets[:, None] >= col_offsets[None, :]
    acc = tl.where(lower_tri_mask, acc, 0.0)

    c_ptrs = C + row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_block)


def triton_matmul_lower_tri(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, N = A.shape[0], B.shape[1]
    K = A.shape[1]
    assert A.shape[1] == B.shape[0], "A and B must have compatible dimensions."
    
    C = torch.empty((M, N), dtype=torch.float32, device='cuda')
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    
    matmul_lower_tri_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul_lower_tri(A, B)


M = 4096

def get_inputs():
    A = torch.rand(M, M, device='cuda')
    B = torch.rand(M, M, device='cuda')
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []