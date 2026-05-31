import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_a0, stride_a1, stride_a2,
    stride_b0, stride_b1, stride_b2,
    stride_c0, stride_c1, stride_c2,
    batch_size, m, k, n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_row = row_offsets < m
    mask_col = col_offsets < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in range(0, k, BLOCK_K):
        k_block_offsets = k_offset + tl.arange(0, BLOCK_K)
        mask_k = k_block_offsets < k

        a_ptrs = (A_ptr +
                  pid_b * stride_a0 +
                  row_offsets[:, None] * stride_a1 +
                  k_block_offsets[None, :] * stride_a2)
        
        b_ptrs = (B_ptr +
                  pid_b * stride_b0 +
                  k_block_offsets[:, None] * stride_b1 +
                  col_offsets[None, :] * stride_b2)

        a_tile = tl.load(a_ptrs, mask=mask_row[:, None] & mask_k[None, :], other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_col[None, :], other=0.0)

        acc += tl.dot(a_tile, b_tile)

    c_ptrs = (C_ptr +
              pid_b * stride_c0 +
              row_offsets[:, None] * stride_c1 +
              col_offsets[None, :] * stride_c2)
    
    mask_c = mask_row[:, None] & mask_col[None, :]
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()

    batch_size, m, k = A.shape
    _, _, n = B.shape

    C = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128

    num_tiles_m = (m + BLOCK_M - 1) // BLOCK_M
    num_tiles_n = (n + BLOCK_N - 1) // BLOCK_N

    grid = (batch_size, num_tiles_m, num_tiles_n)

    bmm_kernel[grid](
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        batch_size, m, k, n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)