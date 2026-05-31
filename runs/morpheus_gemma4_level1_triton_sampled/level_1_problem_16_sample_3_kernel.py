import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Map program IDs to the block of C being computed
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks in A and B
    # A is accessed as A^T, so stride_am is 1 and stride_ak is M_cols (original A's width)
    a_ptr += (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr += (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks from A and B
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < K - k * BLOCK_SIZE_K) & (rn[None, :] < N), other=0.0)

        # Matrix multiplication
        accumulator += tl.dot(a, b)

        # Advance pointers to the next block along K
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # Store the result in C
    c_ptr += (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr, accumulator, mask=c_mask)

def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    # A shape: (K, M), B shape: (K, N)
    # We want result C = A.T @ B, which is (M, N)
    K, M = A.shape
    _K, N = B.shape
    
    assert K == _K, "Inner dimensions must match"
    
    # Ensure tensors are contiguous and on CUDA
    A = A.contiguous()
    B = B.contiguous()
    
    # Output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Strides for A^T: A is (K, M), A^T is (M, K)
    # A^T[m, k] = A[k, m]. 
    # Stride along M (row of A^T) is 1, stride along K (col of A^T) is M.
    stride_am = 1
    stride_ak = M
    
    # Strides for B: (K, N)
    stride_bk = N
    stride_bn = 1
    
    # Strides for C: (M, N)
    stride_cm = N
    stride_cn = 1

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
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
        return triton_matmul(A, B)