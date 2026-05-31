import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, stride_ab,
    stride_bk, stride_bn, stride_bb,
    stride_cm, stride_cn, stride_cb,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    # Compute offsets for the current batch
    a_ptr = a_ptr + pid_batch * stride_ab
    b_ptr = b_ptr + pid_batch * stride_bb
    c_ptr = c_ptr + pid_batch * stride_cb

    # Range of indices for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for A and B tiles
    # A is (M, K), B is (K, N)
    # a_ptr + row * stride_am + col * stride_ak
    # b_ptr + row * stride_bk + col * stride_bn
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_offset in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        # Load tile from A
        # Shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = rm[:, None] * stride_am + (k_offset * BLOCK_SIZE_K + rk[None, :]) * stride_ak
        a = tl.load(a_ptr + a_offsets, mask=(rm[:, None] < m) & ((k_offset * BLOCK_SIZE_K + rk[None, :]) < k), other=0.0)
        
        # Load tile from B
        # Shape: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_offsets = (k_offset * BLOCK_SIZE_K + rk[:, None]) * stride_bk + rn[None, :] * stride_bn
        b = tl.load(b_ptr + b_offsets, mask=((k_offset * BLOCK_SIZE_K + rk[:, None]) < k) & (rn[None, :] < n), other=0.0)
        
        # Matrix multiplication of tiles
        accumulator += tl.dot(a, b)

    # Store the result in C
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptr + c_offsets, accumulator, mask=(rm[:, None] < m) & (rn[None, :] < n))


def triton_bmm(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are on GPU and contiguous
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Dimensions
    batch_size, m, k = a.shape
    _, k_b, n = b.shape
    assert k == k_b, "Inner dimensions must match for matrix multiplication."

    # Output tensor
    c = torch.empty((batch_size, m, n), device=a.device, dtype=a.dtype)

    # Strides
    stride_am, stride_ak, stride_ab = a.stride()
    stride_bk, stride_bn, stride_bb = b.stride()
    stride_cm, stride_cn, stride_cb = c.stride()

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid: (M blocks, N blocks, Batch blocks)
    grid = (
        triton.cdiv(m, BLOCK_SIZE_M),
        triton.cdiv(n, BLOCK_SIZE_N),
        batch_size,
    )

    bmm_kernel[grid](
        a, b, c,
        m, n, k,
        stride_am, stride_ak, stride_ab,
        stride_bk, stride_bn, stride_bb,
        stride_cm, stride_cn, stride_cb,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c


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