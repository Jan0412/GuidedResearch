import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr,  # Pointer to input A: (batch_size, m, k)
    B_ptr,  # Pointer to input B: (batch_size, k, n)
    C_ptr,  # Pointer to output C: (batch_size, m, n)
    batch_size,  # Batch dimension
    m, n, k,  # Matrix dimensions
    stride_ab, stride_am, stride_ak,  # Strides for A
    stride_bb, stride_bk, stride_bn,  # Strides for B
    stride_cb, stride_cm, stride_cn,  # Strides for C
    BLOCK_M: tl.constexpr,  # Block size for M dimension
    BLOCK_N: tl.constexpr,  # Block size for N dimension
    BLOCK_K: tl.constexpr,  # Block size for K dimension
):
    # Batch index
    batch_id = tl.program_id(2)
    # Matrix row index
    pid_m = tl.program_id(0)
    # Matrix column index
    pid_n = tl.program_id(1)

    # Offsets for batch dimension
    offset_batch = batch_id * stride_ab

    # Offsets for M dimension (rows of A and C)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Offsets for N dimension (columns of B and C)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Offsets for K dimension (columns of A and rows of B)
    offsets_k = tl.arange(0, BLOCK_K)

    # Pointers to A and B for this tile
    # A: [batch_id, offsets_m, offsets_k]
    a_ptrs = A_ptr + offset_batch + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    # B: [batch_id, offsets_k, offsets_n]
    b_ptrs = B_ptr + offset_batch + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn

    # Initialize accumulator for C tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in blocks
    for k_block in range(0, tl.cdiv(k, BLOCK_K)):
        # Load A tile
        a = tl.load(a_ptrs, mask=offsets_k[None, :] < k - k_block * BLOCK_K, other=0.0)
        # Load B tile
        b = tl.load(b_ptrs, mask=offsets_k[:, None] < k - k_block * BLOCK_K, other=0.0)
        # Accumulate matrix multiplication
        acc += tl.dot(a, b)
        # Update pointers for next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Convert accumulator to float16 if needed, but we're targeting FP32 so keep as float32
    acc = acc.to(tl.float32)

    # Store the result to C
    # C: [batch_id, offsets_m, offsets_n]
    c_ptrs = C_ptr + offset_batch + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    # Create mask to handle out-of-bounds elements
    mask_m = offsets_m[:, None] < m
    mask_n = offsets_n[None, :] < n
    mask = mask_m & mask_n
    tl.store(c_ptrs, acc, mask=mask)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs batched matrix multiplication using Triton kernel.

    Args:
        A: Input tensor of shape (batch_size, m, k).
        B: Input tensor of shape (batch_size, k, n).

    Returns:
        C: Output tensor of shape (batch_size, m, n).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    batch_size, m, k = A.shape
    _, _, n = B.shape

    # Ensure A and B have the same batch dimension
    assert B.shape[0] == batch_size, "Batch dimensions must match."

    # Prepare output tensor
    C = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)

    # Get strides for the kernel
    stride_ab, stride_am, stride_ak = A.stride()
    stride_bb, stride_bk, stride_bn = B.stride()
    stride_cb, stride_cm, stride_cn = C.stride()

    # Set block sizes for the kernel (tunable parameters)
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    # Determine grid dimensions
    grid = (
        triton.cdiv(m, BLOCK_M),
        triton.cdiv(n, BLOCK_N),
        batch_size,
    )

    # Launch the kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return C


class ModelNew(nn.Module):
    """
    Optimized version of the model that uses a custom Triton kernel for batched matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using optimized Triton kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)