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
    # Map program IDs to the corresponding tile of the output matrix C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current tile
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Base pointers for the current tile in A and B
    # a_ptr is shifted by the row offset in A
    a_base = a_ptr + (rm[:, None] * stride_am)
    # b_ptr is shifted by the column offset in B
    b_base = b_ptr + (rn[None, :] * stride_bn)

    # Initialize accumulator for the result tile
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        
        # Load the tile from A and B
        # Masking ensures we don't read out of bounds if M, N, or K are not multiples of block sizes
        a = tl.load(a_base + (k_start + rk[None, :]) * stride_ak, 
                    mask=(rm[:, None] < M) & (k_start + rk[None, :] < K), other=0.0)
        b = tl.load(b_base + (k_start + rk[:, None]) * stride_bk, 
                    mask=(rn[None, :] < N) & (k_start + rk[:, None] < K), other=0.0)
        
        # Perform matrix multiplication on the tiles
        acc += tl.dot(a, b)

    # Store the final result tile into C
    c_ptr_final = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptr_final, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Grid definition: one program for each tile of the output matrix
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the Triton implementation.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return triton_matmul(A, B)