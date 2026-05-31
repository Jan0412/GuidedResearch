import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def lower_tri_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Optimization: In lower triangular matrix multiplication C = A * B,
    # C[i, j] = 0 if i < j. Thus, we only compute blocks where pid_m >= pid_n.
    if pid_m < pid_n:
        return

    # Ranges for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Pointers to the start of the block in A and B
    # A is (M, M), B is (M, M)
    # A_ptr is accessed as A[rm, rk], B_ptr as B[rk, rn]
    
    # Accumulator for the result tile
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Optimization: The sum C[i, j] = sum_{k=0}^{M-1} A[i, k] * B[k, j]
    # Since A and B are lower triangular:
    # A[i, k] = 0 if k > i
    # B[k, j] = 0 if k < j
    # Therefore, the non-zero range for k is [j, i].
    # For a block (pid_m, pid_n), k must be at least the minimum j in the block
    # and at most the maximum i in the block.
    k_start = pid_n * BLOCK_SIZE_N
    k_end = (pid_m + 1) * BLOCK_SIZE_M

    for k in range(k_start, k_end, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load tiles from A and B
        # Masking is used to handle boundary conditions and ensure we stay within [0, M)
        a = tl.load(A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak), 
                    mask=(rm[:, None] < M) & (rk[None, :] < M), other=0.0)
        b = tl.load(B_ptr + (rk[:, None] * stride_bm + rn[None, :] * stride_bk), 
                    mask=(rk[:, None] < M) & (rn[None, :] < M), other=0.0)
        
        # Perform matrix multiplication on the tiles
        acc += tl.dot(a, b)

    # Store the result tile back to C
    c_mask = (rm[:, None] < M) & (rn[None, :] < M)
    tl.store(C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn), acc, mask=c_mask)


def triton_lower_tri_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wraps the Triton kernel for lower triangular matrix multiplication.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for correct pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    
    M = A.shape[0]
    C = torch.empty((M, M), device=A.device, dtype=torch.float32)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    # Grid dimensions: one block for each tile of the output matrix
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(M, BLOCK_SIZE_N))

    lower_tri_matmul_kernel[grid](
        A, B, C,
        M,
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
    Optimized model that performs matrix multiplication of lower triangular matrices A and B.
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
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        # Ensure inputs are on GPU
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
            
        return triton_lower_tri_matmul(A, B)