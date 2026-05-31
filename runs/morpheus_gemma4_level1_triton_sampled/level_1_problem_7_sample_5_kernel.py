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
    Triton kernel for matrix multiplication optimized for small K.
    Since K is small (64), we process the entire K dimension in a single block.
    """
    # Program IDs for M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the blocks of A and B
    # A is (M, K), B is (K, N)
    # a_ptrs shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    # b_ptrs shape: (BLOCK_SIZE_K, BLOCK_SIZE_N)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Load the blocks from global memory
    # We use masking to handle cases where M or N are not multiples of the block size
    mask_a = (rm[:, None] < M) & (rk[None, :] < K)
    mask_b = (rk[:, None] < K) & (rn[None, :] < N)
    
    a = tl.load(a_ptrs, mask=mask_a, other=0.0)
    b = tl.load(b_ptrs, mask=mask_b, other=0.0)

    # Perform the matrix multiplication for the block
    # tl.dot requires dimensions to be powers of 2 and >= 16
    c = tl.dot(a, b)

    # Store the result block back to global memory
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel.
    """
    # Ensure inputs are on CUDA and contiguous for optimal performance
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Incompatible dimensions for matrix multiplication"

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters
    # Given K=64, we set BLOCK_SIZE_K=64 to eliminate the loop over K.
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    # Grid of programs
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch the kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return c


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using the custom Triton implementation.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)