import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    batch_id = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Pointers to the start of the matrices for the current batch
    a_ptr += batch_id * stride_ab
    b_ptr += batch_id * stride_bb
    c_ptr += batch_id * stride_cb

    # Offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the blocks of A and B
    # A: [M, K], B: [K, N]
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks from A and B
        # Masking for boundaries
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (k * BLOCK_SIZE_K + rk[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k * BLOCK_SIZE_K + rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store the result
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    """
    Triton wrapper for batched matrix multiplication.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, k_b, n = B.shape
    assert k == k_b, "Incompatible dimensions for BMM"

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
    grid = (batch_size, triton.cdiv(m, BLOCK_SIZE_M), triton.cdiv(n, BLOCK_SIZE_N))

    bmm_kernel[grid](
        A, B, C,
        m, n, k,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
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
        Performs batched matrix multiplication using Triton.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)