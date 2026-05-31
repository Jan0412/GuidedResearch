import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tri_matmul_kernel(
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

    # Optimization: Skip blocks that are entirely in the lower triangle.
    # A block is entirely in the lower triangle if the minimum row index 
    # is greater than or equal to the maximum column index.
    if pid_m * BLOCK_M >= (pid_n + 1) * BLOCK_N:
        return

    # Create offsets for the current block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(N, BLOCK_K)):
        rk = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Load blocks of A and B
        # A is upper triangular: A[i, k] = 0 if i > k
        # B is upper triangular: B[k, j] = 0 if k > j
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, 
            mask=(rm[:, None] < N) & (rk[None, :] < N) & (rm[:, None] <= rk[None, :]), 
            other=0.0
        )
        b = tl.load(
            B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn, 
            mask=(rk[:, None] < N) & (rn[None, :] < N) & (rk[:, None] <= rn[None, :]), 
            other=0.0
        )
        
        # Perform matrix multiplication for the block
        acc += tl.dot(a, b)

    # Store the result in C. 
    # Mask the output to ensure only the upper triangle (i <= j) is stored.
    mask_c = (rm[:, None] < N) & (rn[None, :] < N) & (rm[:, None] <= rn[None, :])
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, 
        acc, 
        mask=mask_c
    )

def triton_triangular_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Triton wrapper for upper triangular matrix multiplication.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    C = torch.zeros((N, N), device=A.device, dtype=torch.float32)
    
    # Strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    # Tuning parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Grid: one program per block of the output matrix C
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    tri_matmul_kernel[grid](
        A, B, C,
        N,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
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
        return triton_triangular_matmul(A, B)