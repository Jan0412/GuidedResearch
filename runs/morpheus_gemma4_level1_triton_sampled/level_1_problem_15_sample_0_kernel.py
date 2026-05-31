import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def lower_tri_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block boundaries
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N

    # Optimization: Since A and B are lower triangular, C = A @ B is also lower triangular.
    # If the current block is entirely in the upper triangle (rn >= rm + BLOCK_M), we skip it.
    if rn >= rm + BLOCK_M:
        return

    # Offsets for M and N dimensions
    offs_am = rm + tl.arange(0, BLOCK_M)
    offs_bn = rn + tl.arange(0, BLOCK_N)

    # Accumulator for the result block
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Optimization: For lower triangular matrices, C[i, j] = sum_{k=j}^i A[i, k] * B[k, j]
    # For a block (rm, rn), the k-range needed is [rn, rm + BLOCK_M - 1].
    k_start = (rn // BLOCK_K) * BLOCK_K
    k_end = ((rm + BLOCK_M - 1) // BLOCK_K + 1) * BLOCK_K
    
    # Ensure k indices stay within [0, N)
    k_start = tl.maximum(0, k_start)
    k_end = tl.minimum(N, k_end)

    for k in range(k_start, k_end, BLOCK_K):
        offs_ak = k + tl.arange(0, BLOCK_K)
        offs_bk = k + tl.arange(0, BLOCK_K)

        # Load blocks from A and B
        # A is (N, N), B is (N, N)
        a = tl.load(
            A_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak),
            mask=(offs_am[:, None] < N) & (offs_ak[None, :] < N),
            other=0.0
        )
        b = tl.load(
            B_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn),
            mask=(offs_bk[:, None] < N) & (offs_bn[None, :] < N),
            other=0.0
        )

        # Matrix multiply-accumulate
        accumulator += tl.dot(a, b)

    # Mask for lower triangular result: i >= j
    # offs_am is (BLOCK_M,), offs_bn is (BLOCK_N,)
    mask = offs_am[:, None] >= offs_bn[None, :]
    
    # Store the result
    tl.store(
        C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn),
        accumulator,
        mask=mask & (offs_am[:, None] < N) & (offs_bn[None, :] < N)
    )

def triton_lower_tri_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    C = torch.empty((N, N), device=A.device, dtype=A.dtype)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid: (N/BLOCK_M, N/BLOCK_N)
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))

    lower_tri_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of lower triangular matrices A and B.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B using a custom Triton kernel.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N), which is lower triangular.
        """
        # The Triton kernel handles the lower triangular property and the tril operation.
        return triton_lower_tri_matmul(A, B)