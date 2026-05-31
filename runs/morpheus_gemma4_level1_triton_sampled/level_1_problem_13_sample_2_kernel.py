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
    """
    Triton kernel for matrix multiplication C = A * B.
    Optimized for FP32 precision.
    """
    # Map program IDs to the corresponding block of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the first block of A and B
    # A is (M, K), B is (K, N)
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize the accumulator in FP32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the inner dimension K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        # Masking is used to handle cases where M, N, or K are not multiples of BLOCK_SIZE
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        # Perform the dot product and accumulate
        accumulator += tl.dot(a, b)

        # Advance pointers to the next block along the K dimension
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Pointers to the output block of C
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    
    # Store the final result
    tl.store(c_ptrs, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for efficient memory access
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match for matrix multiplication."

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tunable block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Grid definition: one program for each block of the output matrix C
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    # Launch the kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices using Triton.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return triton_matmul(A, B)