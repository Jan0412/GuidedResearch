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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication of two upper triangular matrices.
    The product of two upper triangular matrices is also upper triangular.
    C[i, j] = sum_{k=i}^j A[i, k] * B[k, j]
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # If the block is entirely in the lower triangle, we can skip it.
    # Since BLOCK_M == BLOCK_N, pid_m > pid_n means the block is below the diagonal.
    if pid_m > pid_n:
        return

    # Create offsets for the block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # The range of k that contributes to C[i, j] is [i, j].
    # For a block (pid_m, pid_n), the minimum i is pid_m * BLOCK_M 
    # and the maximum j is (pid_n + 1) * BLOCK_N.
    k_start = pid_m * BLOCK_M
    k_end = (pid_n + 1) * BLOCK_N
    
    # Align k_start and k_end to BLOCK_K
    k_start_block = k_start // BLOCK_K
    k_end_block = (k_end + BLOCK_K - 1) // BLOCK_K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the reduction dimension K
    for k_block in range(k_start_block, k_end_block):
        rk = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Load A block: A[rm, rk]
        # Masking for A: A is upper triangular, but we already restricted the k-loop.
        # We still need to mask for the boundaries of the matrix N.
        a_mask = (rm[:, None] < N) & (rk[None, :] < N)
        a = tl.load(A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, mask=a_mask, other=0.0)
        
        # Load B block: B[rk, rn]
        b_mask = (rk[:, None] < N) & (rn[None, :] < N)
        b = tl.load(B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn, mask=b_mask, other=0.0)
        
        # Dot product
        acc += tl.dot(a, b)

    # Final mask for the output C: only store the upper triangular part (i <= j)
    c_mask = (rm[:, None] < N) & (rn[None, :] < N) & (rm[:, None] <= rn[None, :])
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=c_mask)


def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton upper triangular matrix multiplication kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must be square and of the same size."
    
    N = A.shape[0]
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.zeros((N, N), device=A.device, dtype=torch.float32)

    # Tunable block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid dimensions
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))

    triu_matmul_kernel[grid](
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
    Optimized model that performs matrix multiplication (C = A * B) for upper triangular matrices
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_triu_matmul(A, B)