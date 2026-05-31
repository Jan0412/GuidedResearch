import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triu_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Optimization: If the block is entirely in the lower triangle of the result, skip it.
    if pid_m > pid_n:
        return

    # Define the row and column indices for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # For upper triangular multiplication C = A * B:
    # C[i, j] = sum_{k=i to j} A[i, k] * B[k, j]
    # For a block (pid_m, pid_n), the relevant k range is [min(i), max(j)]
    k_start = pid_m * BLOCK_SIZE_M
    k_end = (pid_n + 1) * BLOCK_SIZE_N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension in blocks
    for k in range(k_start, k_end, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load block from A: A[rm, rk]
        # Masking ensures we stay within bounds of the matrix N
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < N) & (rk[None, :] < N),
            other=0.0
        )
        
        # Load block from B: B[rk, rn]
        b = tl.load(
            B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(rk[:, None] < N) & (rn[None, :] < N),
            other=0.0
        )
        
        # Perform matrix multiplication for the block
        acc += tl.dot(a, b)

    # Mask the final output to ensure result is upper triangular (i <= j)
    # This is critical for the diagonal block where pid_m == pid_n
    mask = (rm[:, None] < N) & (rn[None, :] < N) & (rm[:, None] <= rn[None, :])
    
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=mask,
        other=0.0
    )

def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Triton wrapper for upper triangular matrix multiplication.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    C = torch.zeros((N, N), device=A.device, dtype=torch.float32)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid: divide the output matrix into blocks
    grid = (triton.cdiv(N, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    triu_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using Triton.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_triu_matmul(A, B)