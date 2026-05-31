import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matvec_kernel(
    A_ptr,  # Pointer to input matrix A
    B_ptr,  # Pointer to input vector B
    C_ptr,  # Pointer to output vector C
    M,      # Number of rows in A
    K,      # Number of columns in A / elements in B
    BLOCK_SIZE_K: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    acc = 0.0

    for k in range(0, K, BLOCK_SIZE_K):
        mask_k = offsets_k < (K - k)
        A_block = tl.load(A_ptr + row_idx * K + k + offsets_k, mask=mask_k, other=0.0)
        B_block = tl.load(B_ptr + k + offsets_k, mask=mask_k, other=0.0)
        acc += tl.sum(A_block * B_block)

    tl.store(C_ptr + row_idx, acc)


def triton_matvec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    C = torch.empty((M, 1), dtype=torch.float32, device='cuda')

    BLOCK_SIZE_K = 1024
    grid = (M,)
    matvec_kernel[grid](A, B, C, M, K, BLOCK_SIZE_K=BLOCK_SIZE_K)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matvec(A, B)