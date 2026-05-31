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
    # Map program ID to the row and column of the output matrix C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Initialize the accumulator for the dot product
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Compute pointers for the current block of A and B
        # A is (M, K), B is (K, N)
        a_off = (rm[:, None] * stride_am + (k * BLOCK_K + rk[None, :]) * stride_ak)
        b_off = ((k * BLOCK_K + rk[:, None]) * stride_bk + (pid_n * BLOCK_N + rn[None, :]) * stride_bn)

        # Load blocks with boundary masking
        a = tl.load(a_ptr + a_off, mask=(rm[:, None] < M) & ((k * BLOCK_K + rk[None, :]) < K), other=0.0)
        b = tl.load(b_ptr + b_off, mask=((k * BLOCK_K + rk[:, None]) < K) & (pid_n * BLOCK_N + rn[None, :] < N), other=0.0)

        # Perform the matrix multiplication for the block
        acc += tl.dot(a, b)

    # Store the final result in the output matrix C
    c_off = (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptr + c_off, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure tensors are contiguous and on GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Incompatible dimensions"

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters: Since K is small (64), we use a BLOCK_K that fits K.
    # M and N are large, so we tile them.
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64 

    # Grid is defined by the number of blocks needed to cover M and N
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

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
    Optimized model that performs matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using a Triton-optimized kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)