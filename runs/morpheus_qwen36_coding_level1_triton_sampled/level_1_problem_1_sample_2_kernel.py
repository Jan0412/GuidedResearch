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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = M // BLOCK_SIZE_M
    num_pid_n = N // BLOCK_SIZE_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    A_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bn + offs_bn[None, :] * stride_cn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_iters = K // BLOCK_SIZE_K
    for k in range(num_iters):
        A = tl.load(A_ptrs, mask=offs_k[None, :] < BLOCK_SIZE_K, other=0.0)
        B = tl.load(B_ptrs, mask=offs_k[:, None] < BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(A, B, allow_tf32=False)
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bn

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    C_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_ptrs, accumulator, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matmul."

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match."

    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    num_pid_m = M // BLOCK_SIZE_M
    num_pid_n = N // BLOCK_SIZE_N
    num_blocks = num_pid_m * num_pid_n
    grid = (num_blocks, 1, 1)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)


N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N, device="cuda")
    B = torch.rand(N, N, device="cuda")
    return [A, B]

def get_init_inputs():
    return []