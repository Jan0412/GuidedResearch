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
    Optimized for cases where the inner dimension K is small (tall-skinny matrices).
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets for the output matrix C
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Accumulator for the result block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the inner dimension K in chunks of BLOCK_SIZE_K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute pointers for the current K-block
        # A block shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # B block shape: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        a_offset = (rm[:, None] * stride_am + (k * BLOCK_SIZE_K + rk[None, :]) * stride_ak)
        b_offset = ((k * BLOCK_SIZE_K + rk[:, None]) * stride_bk + rn[None, :] * stride_bn)
        
        # Load the blocks with boundary masking
        a = tl.load(a_ptr + a_offset, mask=(rm[:, None] < M) & ((k * BLOCK_SIZE_K + rk[None, :]) < K), other=0.0)
        b = tl.load(b_ptr + b_offset, mask=((k * BLOCK_SIZE_K + rk[:, None]) < K) & (rn[None, :] < N), other=0.0)
        
        # Compute the dot product for the current block
        acc += tl.dot(a, b)

    # Store the final result block back to memory
    c_offset = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptr + c_offset, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

def triton_matmul(a, b):
    """
    Wrapper function to launch the Triton matmul kernel.
    """
    # a: (M, K), b: (K, N)
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"
    
    # Ensure inputs are contiguous on GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    
    # Initialize output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Tunable block sizes. 
    # Given M=32768, N=32768, K=32, we set BLOCK_SIZE_K=32 to process the inner dimension in one step.
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid: Number of blocks needed to cover the output matrix C
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
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
    Optimized model that performs matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication A * B.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K).
            B (torch.Tensor): Input matrix of shape (K, N).

        Returns:
            torch.Tensor: Output matrix of shape (M, N).
        """
        return triton_matmul(A, B)