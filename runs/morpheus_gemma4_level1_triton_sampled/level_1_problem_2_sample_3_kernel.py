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
    # Map program IDs to the corresponding block of the output matrix C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the blocks of A and B
    # a_ptrs shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)
    # b_ptrs shape: (BLOCK_SIZE_K, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator for the dot product in FP32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in blocks of BLOCK_SIZE_K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles from A and B with boundary masking
        a = tl.load(
            a_ptrs + (k * BLOCK_SIZE_K * stride_ak), 
            mask=(rm[:, None] < M) & ((k * BLOCK_SIZE_K + rk[None, :]) < K), 
            other=0.0
        )
        b = tl.load(
            b_ptrs + (k * BLOCK_SIZE_K * stride_bk), 
            mask=((k * BLOCK_SIZE_K + rk[:, None]) < K) & (rn[None, :] < N), 
            other=0.0
        )
        
        # Perform matrix multiplication on the blocks and accumulate
        accumulator += tl.dot(a, b)

    # Store the final result in matrix C with boundary masking
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match for matrix multiplication."

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Get strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid definition
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
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
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using the Triton implementation.

        Args:
            A: Input tensor of shape (M, K) on CUDA.
            B: Input tensor of shape (K, N) on CUDA.

        Returns:
            Output tensor of shape (M, N) on CUDA.
        """
        return triton_matmul(A, B)