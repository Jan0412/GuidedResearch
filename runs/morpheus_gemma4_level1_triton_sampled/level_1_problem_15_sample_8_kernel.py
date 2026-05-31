import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tril_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
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

    # Since A and B are lower triangular, the result C = A @ B is also lower triangular.
    # We only compute blocks where the row index is greater than or equal to the column index.
    if pid_m < pid_n:
        return

    # Define the ranges for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the first block of A and B
    # A is lower triangular: A[i, k] = 0 if k > i
    # B is lower triangular: B[k, j] = 0 if k < j
    # Therefore, for a block (pid_m, pid_n), the range of k that contributes is [pid_n * BN, (pid_m + 1) * BM)
    k_start = pid_n * BLOCK_SIZE_N
    k_end = (pid_m + 1) * BLOCK_SIZE_M
    
    # Ensure k_start and k_end are within [0, N]
    k_start = tl.maximum(0, k_start)
    k_end = tl.minimum(N, k_end)

    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the k dimension
    for k in range(k_start, k_end, BLOCK_SIZE_K):
        # Load blocks of A and B
        # A pointer: a_ptr + rm * stride_am + rk * stride_ak
        # B pointer: b_ptr + rk * stride_bk + rn * stride_bn
        a = tl.load(
            a_ptr + (rm[:, None] * stride_am + (k + rk[None, :]) * stride_ak),
            mask=(rm[:, None] < N) & ((k + rk[None, :]) < N),
            other=0.0
        )
        b = tl.load(
            b_ptr + ((k + rk[:, None]) * stride_bk + rn[None, :] * stride_bn),
            mask=((k + rk[:, None]) < N) & (rn[None, :] < N),
            other=0.0
        )
        
        # Matrix multiplication
        acc += tl.dot(a, b)

    # Store the result
    # Mask to ensure we only store the lower triangular part (i >= j)
    # This is especially important for the diagonal blocks (pid_m == pid_n)
    mask = (rm[:, None] < N) & (rn[None, :] < N) & (rm[:, None] >= rn[None, :])
    tl.store(
        c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn),
        acc,
        mask=mask,
        other=0.0
    )

def triton_tril_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.shape == B.shape, "Matrices must be square and of the same size"
    
    N = A.shape[0]
    A = A.contiguous()
    B = B.contiguous()
    
    # Output tensor
    C = torch.zeros((N, N), device=A.device, dtype=torch.float32)
    
    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Grid definition
    grid = (triton.cdiv(N, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    tril_matmul_kernel[grid](
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
    Optimized model that performs a matrix multiplication of lower triangular matrices
    using a custom Triton kernel to skip upper triangular computations.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.
        """
        # Ensure inputs are in FP32 for the kernel
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        return triton_tril_matmul(A, B)