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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Grid: (batch, m_blocks, n_blocks)
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Offsets for the current block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the current batch's matrices
    a_ptr = A_ptr + pid_batch * stride_ab
    b_ptr = B_ptr + pid_batch * stride_bb
    c_ptr = C_ptr + pid_batch * stride_cb

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles from A and B
        # A: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # B: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        a = tl.load(
            a_ptr + (offs_am[:, None] * stride_am + (k * BLOCK_SIZE_K + offs_k[None, :]) * stride_ak),
            mask=(offs_am[:, None] < M) & ((k * BLOCK_SIZE_K + offs_k[None, :]) < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + ((k * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn),
            mask=((k * BLOCK_SIZE_K + offs_k[:, None]) < K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        
        # Matrix multiplication of tiles
        accumulator += tl.dot(a, b)

    # Store the result back to C
    c_offsets = offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(
        c_ptr + c_offsets,
        accumulator,
        mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N),
    )

def triton_bmm(A: torch.Tensor, B: torch.Tensor):
    # Ensure inputs are contiguous on GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    batch_size, m, k = A.shape
    _, k_b, n = B.shape
    assert k == k_b, "Inner dimensions must match"

    # Output tensor
    C = torch.empty((batch_size, m, n), device=A.device, dtype=A.dtype)

    # Strides
    stride_ab = A.stride(0)
    stride_am = A.stride(1)
    stride_ak = A.stride(2)

    stride_bb = B.stride(0)
    stride_bk = B.stride(1)
    stride_bn = B.stride(2)

    stride_cb = C.stride(0)
    stride_cm = C.stride(1)
    stride_cn = C.stride(2)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid definition
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
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm(A, B)