import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, m, k, n,
    stride_a0, stride_a1, stride_a2,
    stride_b0, stride_b1, stride_b2,
    stride_c0, stride_c1, stride_c2,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Base pointers for the current batch
    a_ptr = A_ptr + pid_batch * stride_a0
    b_ptr = B_ptr + pid_batch * stride_b0
    c_ptr = C_ptr + pid_batch * stride_c0

    # Row and column offsets for the current tile
    row_off = pid_m * BLOCK_M
    col_off = pid_n * BLOCK_N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for start_k in range(0, k, BLOCK_K):
        # Compute offsets for A and B
        offs_a = (row_off + tl.arange(BLOCK_M))[:, None] * stride_a1 + (start_k + tl.arange(BLOCK_K))[None, :] * stride_a2
        offs_b = (start_k + tl.arange(BLOCK_K))[:, None] * stride_b1 + (col_off + tl.arange(BLOCK_N))[None, :] * stride_b2

        # Create masks
        mask_a = (row_off + tl.arange(BLOCK_M))[:, None] < m
        mask_b = (col_off + tl.arange(BLOCK_N))[None, :] < n
        # K dimension masks (though loop bounds usually handle this, good for safety)
        mask_a = mask_a & ((start_k + tl.arange(BLOCK_K))[None, :] < k)
        mask_b = mask_b & ((start_k + tl.arange(BLOCK_K))[:, None] < k)

        # Load tiles
        a = tl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
        b = tl.load(b_ptr + offs_b, mask=mask_b, other=0.0)

        # Matrix multiply
        acc += tl.dot(a, b)

    # Store result
    offs_c = (row_off + tl.arange(BLOCK_M))[:, None] * stride_c1 + (col_off + tl.arange(BLOCK_N))[None, :] * stride_c2
    mask_c = (row_off + tl.arange(BLOCK_M))[:, None] < m
    mask_c = mask_c & ((col_off + tl.arange(BLOCK_N))[None, :] < n)
    tl.store(c_ptr + offs_c, acc, mask=mask_c)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.is_contiguous() and B.is_contiguous(), "Tensors must be contiguous."
    
    batch_size, m, k = A.shape
    k_b, n = B.shape
    assert k == k_b, "Inner dimensions must match."

    # Allocate output
    C = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)

    # Strides
    stride_a0, stride_a1, stride_a2 = A.stride()
    stride_b0, stride_b1, stride_b2 = B.stride()
    stride_c0, stride_c1, stride_c2 = C.stride()

    # Block sizes
    BLOCK_M = 128
    BLOCK_K = 32
    BLOCK_N = 128

    # Grid calculation
    grid = lambda meta: (
        batch_size,
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
    )

    # Launch kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, k, n,
        stride_a0, stride_a1, stride_a2,
        stride_b0, stride_b1, stride_b2,
        stride_c0, stride_c1, stride_c2,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )
    return C


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    Optimized with custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using custom Triton kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)