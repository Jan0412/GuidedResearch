import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_skinny_kernel(
    a_ptr,  # Pointer to matrix A
    b_ptr,  # Pointer to matrix B
    c_ptr,  # Pointer to output matrix C
    M, N, K, # Dimensions: A(M, K), B(K, N), C(M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Triton kernel optimized for cases where K (the inner dimension) is small.
    It computes the matrix multiplication C = A @ B by iterating over the small 
    dimension K and performing outer product updates.
    """
    # Program IDs for the M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create ranges for the current block
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)

    # Offsets for A and B
    # a_ptr is (M, K), we want a slice of size (BLOCK_M, 1) for a specific k
    a_offsets = (pid_m * BLOCK_M + rm) * K
    # b_ptr is (K, N), we want a slice of size (1, BLOCK_N) for a specific k
    b_offsets = pid_n * BLOCK_N + rn

    # Initialize accumulator for the output block C (BLOCK_M, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Since K is small (tall-skinny case), we iterate through K directly.
    # This avoids the overhead of complex tiling in the K dimension.
    for k in range(0, K):
        # Load a column of A for the current block of M: shape (BLOCK_M,)
        a = tl.load(a_ptr + a_offsets + k)
        # Load a row of B for the current block of N: shape (BLOCK_N,)
        b = tl.load(b_ptr + k * N + b_offsets)
        
        # Compute outer product and accumulate: (BLOCK_M, 1) * (1, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += a[:, None] * b[None, :]

    # Calculate output offsets for matrix C: shape (BLOCK_M, BLOCK_N)
    c_offsets = (pid_m * BLOCK_M + rm)[:, None] * N + (pid_n * BLOCK_N + rn)[None, :]
    
    # Store the final result
    tl.store(c_ptr + c_offsets, acc)


def triton_matmul_skinny(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel.
    """
    # Ensure tensors are on GPU and contiguous
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_check, N = b.shape
    assert K == K_check, "Inner dimensions must match"

    # Prepare output tensor
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Tuning parameters
    BLOCK_M = 128
    BLOCK_N = 128

    # Grid: one program per block of M and N
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    # Launch kernel
    matmul_skinny_kernel[grid](
        a, b, out,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication (C = A * B)
    using a custom Triton kernel optimized for tall-skinny matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using a custom Triton kernel.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K).
            B (torch.Tensor): Input matrix of shape (K, N).

        Returns:
            torch.Tensor: Output matrix of shape (M, N).
        """
        # Ensure inputs are on CUDA for the Triton kernel
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
            
        return triton_matmul_skinny(A, B)