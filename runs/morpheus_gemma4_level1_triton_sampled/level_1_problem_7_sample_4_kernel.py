import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the blocks of A and B
    # A is (M, K), B is (K, N)
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Mask for boundaries
    mask_a = (rm[:, None] < M) & (rk[None, :] < K)
    mask_b = (rk[:, None] < K) & (rn[None, :] < N)

    # Load blocks
    a = tl.load(a_ptrs, mask=mask_a, other=0.0)
    b = tl.load(b_ptrs, mask=mask_b, other=0.0)

    # Perform matrix multiplication
    # tl.dot requires inputs to be at least 16x16 and powers of 2
    accumulator = tl.dot(a, b)

    # Store the result
    out_ptrs = out_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=mask_out)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure tensors are contiguous and on GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Output tensor
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters
    # Since K is small (64), we can process the entire K dimension in one block
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = triton.next_power_of_2(K)

    # Grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A * B) 
    using a custom Triton kernel optimized for small K dimensions.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using a Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)