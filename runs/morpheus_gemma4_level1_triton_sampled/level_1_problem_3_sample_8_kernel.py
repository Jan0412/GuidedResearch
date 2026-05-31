import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the batch
    # A is (batch, m, k), B is (batch, k, n), C is (batch, m, n)
    a_batch_ptr = A_ptr + pid_batch * stride_ab
    b_batch_ptr = B_ptr + pid_batch * stride_bb
    c_batch_ptr = C_ptr + pid_batch * stride_cb

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load block from A: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # a_ptr = a_batch_ptr + row_offset * stride_am + col_offset * stride_ak
        a_offsets = rm[:, None] * stride_am + (k * BLOCK_SIZE_K + rk)[None, :] * stride_ak
        a = tl.load(a_batch_ptr + a_offsets, mask=(rm[:, None] < M) & ((k * BLOCK_SIZE_K + rk)[None, :] < K), other=0.0)

        # Load block from B: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # b_ptr = b_batch_ptr + row_offset * stride_bk + col_offset * stride_bn
        b_offsets = (k * BLOCK_SIZE_K + rk)[:, None] * stride_bk + rn[None, :] * stride_bn
        b = tl.load(b_batch_ptr + b_offsets, mask=((k * BLOCK_SIZE_K + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)

        # Matrix multiplication
        accumulator += tl.dot(a, b)

    # Store the result in C
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_batch_ptr + c_offsets, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    """
    Triton implementation of batched matrix multiplication.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    batch_size_b, k_b, n = B.shape
    assert batch_size == batch_size_b and k == k_b, "Batch size and inner dimensions must match."

    # Output tensor
    C = torch.empty((batch_size, m, n), device=A.device, dtype=A.dtype)

    # Strides
    stride_ab, stride_am, stride_ak = A.stride()
    stride_bb, stride_bk, stride_bn = B.stride()
    stride_cb, stride_cm, stride_cn = C.stride()

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid: (batch, m_blocks, n_blocks)
    grid = (batch_size, tl.cdiv(m, BLOCK_SIZE_M), tl.cdiv(n, BLOCK_SIZE_N))

    bmm_kernel[grid](
        A, B, C,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        m, n, k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)