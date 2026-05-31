import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    A_ptr,
    B_ptr,
    out_ptr,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        A_offsets = row_offsets[:, None] * K + col_offsets[None, :] + k
        B_offsets = col_offsets + k

        A_tile = tl.load(A_ptr + A_offsets)
        B_tile = tl.load(B_ptr + B_offsets)[:, None]

        acc += tl.dot(A_tile, B_tile)

    tl.store(out_ptr + row_offsets, acc)


def triton_gemv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    out = torch.empty((M, 1), dtype=A.dtype, device=A.device)

    BLOCK_M = 64
    BLOCK_K = 128

    grid = (M // BLOCK_M,)
    gemv_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        out_ptr=out,
        M=M,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_gemv(A, B)