import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_ak, stride_am,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for C = A^T * B
    A is (K, M), B is (K, N), C is (M, N)
    """
    # Map program IDs to the output matrix C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to blocks of A and B
    # A is accessed as (k, m) to simulate A^T (m, k)
    a_ptr_block = a_ptr + (rk[:, None] * stride_ak + rm[None, :] * stride_am)
    b_ptr_block = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Shift pointers for the current K-block
        a_ptr_curr = a_ptr_block + k * BLOCK_SIZE_K * stride_ak
        b_ptr_curr = b_ptr_block + k * BLOCK_SIZE_K * stride_bk

        # Load blocks with masking for boundary conditions
        # a_block: (BLOCK_SIZE_K, BLOCK_SIZE_M)
        # b_block: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        a_block = tl.load(a_ptr_curr, mask=(rk[:, None] + k * BLOCK_SIZE_K < K) & (rm[None, :] < M), other=0.0)
        b_block = tl.load(b_ptr_curr, mask=(rk[:, None] + k * BLOCK_SIZE_K < K) & (rn[None, :] < N), other=0.0)

        # Compute dot product: A^T_block (M, K) @ B_block (K, N)
        accumulator += tl.dot(tl.trans(a_block), b_block)

    # Store the result in C
    c_ptr_block = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptr_block, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_matmul_transpose_a(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton kernel to compute A.T @ B.
    A: (K, M)
    B: (K, N)
    Returns: (M, N)
    """
    # Ensure inputs are contiguous on GPU
    A = A.contiguous()
    B = B.contiguous()
    
    K, M = A.shape
    K_B, N = B.shape
    assert K == K_B, "Dimension mismatch between A and B"

    # Allocate output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid for the kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A^T * B) using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication A.T @ B.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Use the custom Triton implementation instead of torch.matmul(A.T, B)
        return triton_matmul_transpose_a(A, B)