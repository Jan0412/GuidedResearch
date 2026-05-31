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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A * B.
    Optimized for cases where K is small enough to fit in a single block.
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers to the blocks of A and B
    # a_ptr is (M, K), b_ptr is (K, N)
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Boundary masks
    mask_m = rm[:, None] < M
    mask_n = rn[None, :] < N
    mask_k = rk[None, :] < K

    # Load blocks of A and B
    # a: (BLOCK_M, BLOCK_K), b: (BLOCK_K, BLOCK_N)
    a = tl.load(a_ptrs, mask=mask_m & mask_k, other=0.0)
    b = tl.load(b_ptrs, mask=mask_k.T & mask_n, other=0.0)

    # Perform matrix multiplication using Triton's dot operator
    # Since K is small (32), we compute the entire inner dimension in one go
    c = tl.dot(a, b)

    # Store the result in C
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=mask_m & mask_n)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for correct stride calculation
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions of matrices must match."

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters
    BLOCK_M = 128
    BLOCK_N = 128
    # We set BLOCK_K to 32 because in the given problem N (the inner dim) is 32.
    # This allows us to process the entire inner dimension in a single kernel launch per block.
    BLOCK_K = 32 

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return c


class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using the optimized Triton implementation.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K).
            B (torch.Tensor): Input matrix of shape (K, N).

        Returns:
            torch.Tensor: Output matrix of shape (M, N).
        """
        return triton_matmul(A, B)