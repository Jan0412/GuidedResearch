import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    # L2 cache optimization: Grouping
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = num_pid_m * (pid_m % (num_pid_m // GROUP_SIZE_M)) if num_pid_m >= GROUP_SIZE_M else pid_m
    # Simplified grouping for BMM: we primarily group along M
    # To keep it robust, we'll use a simpler mapping or standard tiling
    # Re-calculating pid_m for the actual block
    # For simplicity and correctness in a general BMM, we use the basic mapping:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for the first block
    # A is (batch, M, K), B is (batch, K, N)
    # A_ptr offset = batch * (M*K) + row * K + col
    # B_ptr offset = batch * (K*N) + row * N + col
    
    a_ptr = A_ptr + pid_batch * stride_ab + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr = B_ptr + pid_batch * stride_bb + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        # Dot product
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # Store result
    c_ptr = C_ptr + pid_batch * stride_cb + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptr, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N) )

def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, k_b, n = B.shape
    assert k == k_b, "Inner dimensions must match"

    C = torch.empty((batch_size, m, n), device=A.device, dtype=A.dtype)

    # Strides
    stride_ab, stride_am, stride_ak = A.stride()
    stride_bb, stride_bk, stride_bn = B.stride()
    stride_cb, stride_cm, stride_cn = C.stride()

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Grid: (M_blocks, N_blocks, Batch_blocks)
    grid = (
        triton.cdiv(m, BLOCK_SIZE_M),
        triton.cdiv(n, BLOCK_SIZE_N),
        batch_size,
    )

    bmm_kernel[grid](
        A, B, C,
        m, n, k,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
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
        # Ensure tensors are on GPU and in FP32
        A = A.cuda().float()
        B = B.cuda().float()
        return triton_bmm(A, B)