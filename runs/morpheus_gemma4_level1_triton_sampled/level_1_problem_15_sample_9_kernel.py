import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triangular_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for multiplying two lower triangular matrices.
    C = tril(A @ B)
    Since A and B are lower triangular, C is also lower triangular.
    C[i, j] = sum_{k=0 to min(i, j)} A[i, k] * B[k, j]
    For i >= j, C[i, j] = sum_{k=0 to j} A[i, k] * B[k, j]
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block ranges
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Optimization: If the entire block is in the upper triangle (i < j), skip it.
    # Block is in upper triangle if max(rm) < min(rn)
    if (pid_m + 1) * BLOCK_SIZE_M <= pid_n * BLOCK_SIZE_N:
        return

    # Pointers to the start of the blocks
    # a_ptr is (M, K), b_ptr is (K, N)
    a_ptr += (pid_m * BLOCK_SIZE_M * stride_am)
    b_ptr += (pid_n * BLOCK_SIZE_N * stride_bn)

    # Accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # The inner loop over K. 
    # Since B is lower triangular, B[k, j] = 0 if k > j.
    # For any j in the current block [pid_n * BN, (pid_n + 1) * BN), 
    # the maximum k that can contribute is (pid_n + 1) * BN - 1.
    k_limit = (pid_n + 1) * BLOCK_SIZE_N
    
    for k in range(0, tl.minimum(K, k_limit), BLOCK_SIZE_K):
        # Load blocks of A and B
        # A block: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        # B block: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        a = tl.load(a_ptr + (rm[:, None] * stride_am + (k + rk[None, :]) * stride_ak), 
                    mask=(rm[:, None] < M) & ((k + rk[None, :]) < K), other=0.0)
        b = tl.load(b_ptr + ((k + rk[:, None]) * stride_bk + rn[None, :] * stride_bn), 
                    mask=((k + rk[:, None]) < K) & (rn[None, :] < N), other=0.0)
        
        # Dot product
        accumulator += tl.dot(a, b)

    # Mask for the boundaries of the matrix and the lower triangular part (i >= j)
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    mask_tri = rm[:, None] >= rn[None, :]
    final_mask = mask_out & mask_tri

    # Store the result
    c_ptr += (pid_m * BLOCK_SIZE_M * stride_cm)
    tl.store(c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn), 
             accumulator, mask=final_mask, other=0.0)


def triton_triangular_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton triangular matrix multiplication kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_B, N = B.shape
    assert K == K_B, "Inner dimensions must match."

    # Output tensor initialized to zero to handle the upper triangle
    C = torch.zeros((M, N), device=A.device, dtype=A.dtype)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid definition
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    triangular_matmul_kernel[grid](
        A, B, C,
        M, N, K,
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
    using a custom Triton kernel to avoid computing the upper triangle.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        # Ensure inputs are on GPU
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
            
        return triton_triangular_matmul(A, B)