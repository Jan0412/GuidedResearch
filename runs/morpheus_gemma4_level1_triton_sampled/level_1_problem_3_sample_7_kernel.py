import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_batch_a, stride_batch_b, stride_batch_c,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute offsets for the current batch
    a_ptr = a_ptr + pid_batch * stride_batch_a
    b_ptr = b_ptr + pid_batch * stride_batch_b
    c_ptr = c_ptr + pid_batch * stride_batch_c

    # Offsets for the M and N blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the blocks of A and B
    # a_ptr is (M, K), b_ptr is (K, N)
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Accumulator for the result
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        # Masking to handle cases where K is not a multiple of BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=(rk[None, :] + k * BLOCK_SIZE_K < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k * BLOCK_SIZE_K < K), other=0.0)
        
        # Matrix multiplication of blocks
        accumulator += tl.dot(a, b)

        # Advance pointers to the next block along K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Compute offsets for the output C
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_ptrs = c_ptr + c_offsets

    # Store the result with masking for M and N
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)

def triton_bmm(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are contiguous and on GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()

    batch_size, m, k = a.shape
    _, k_b, n = b.shape
    assert k == k_b, "Inner dimensions must match"

    # Output tensor
    c = torch.empty((batch_size, m, n), device=a.device, dtype=a.dtype)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Strides
    stride_am, stride_ak = a.stride(1), a.stride(2)
    stride_bk, stride_bn = b.stride(1), b.stride(2)
    stride_cm, stride_cn = c.stride(1), c.stride(2)
    stride_batch_a, stride_batch_b, stride_batch_c = a.stride(0), b.stride(0), c.stride(0)

    # Grid: (batch, m_blocks, n_blocks)
    grid = (batch_size, triton.cdiv(m, BLOCK_SIZE_M), triton.cdiv(n, BLOCK_SIZE_N))

    bmm_kernel[grid](
        a, b, c,
        m, n, k,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_batch_a, stride_batch_b, stride_batch_c,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
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