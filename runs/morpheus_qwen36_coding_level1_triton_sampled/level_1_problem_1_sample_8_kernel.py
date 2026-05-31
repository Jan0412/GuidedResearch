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

    offsets_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    mask_m = offsets_am < M
    mask_n = offsets_bn < N
    mask_k = offsets_k < K

    a_ptrs = A_ptr + offsets_am[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = B_ptr + offsets_k[:, None] * stride_bn + offsets_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bn

    offsets_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions for matrix multiplication"

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    num_tiles_m = (M + BLOCK_M - 1) // BLOCK_M
    num_tiles_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_tiles_m, num_tiles_n, 1)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


N = 2048 * 2


def get_inputs():
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]


def get_init_inputs():
    return []