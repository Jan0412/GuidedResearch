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

    # Create offsets for the blocks
    offs_am = (pid_m * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = (pid_n * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize the accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension to compute the dot product
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute pointers for the current blocks of A and B
        # a_ptr points to (M, K), b_ptr points to (K, N)
        a_offs = offs_am[:, None] * stride_am + (k * BLOCK_SIZE_K + offs_k[None, :]) * stride_ak
        b_offs = (k * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn

        # Load blocks from A and B with boundary masking
        a = tl.load(a_ptr + a_offs, 
                    mask=(offs_am[:, None] < M) & ((k * BLOCK_SIZE_K + offs_k[None, :]) < K), 
                    other=0.0)
        b = tl.load(b_ptr + b_offs, 
                    mask=((k * BLOCK_SIZE_K + offs_k[:, None]) < K) & (offs_bn[None, :] < N), 
                    other=0.0)
        
        # Perform the dot product and accumulate
        accumulator += tl.dot(a, b)

    # Store the final accumulated result in the output matrix C
    c_offs = offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptr + c_offs, accumulator, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Triton-based matrix multiplication wrapper.
    """
    # Ensure tensors are contiguous and on GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Matrix dimensions must match for multiplication."

    # Prepare output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters for block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Define the grid of programs
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch the Triton kernel
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
    Optimized model that performs a square matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using a custom Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return triton_matmul(A, B)