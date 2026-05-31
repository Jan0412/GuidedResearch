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
    Triton kernel for Matrix Multiplication (C = A * B).
    """
    # Map program IDs to the block of C being computed
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current block of C
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize the accumulator for the dot product result
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in blocks of BLOCK_SIZE_K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Load a block of A: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # Masking ensures we don't access memory outside the tensor boundaries
        a = tl.load(
            a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0
        )
        
        # Load a block of B: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn),
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0
        )
        
        # Perform the dot product for the block and accumulate
        accumulator += tl.dot(a, b)

    # Store the final result block into C
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn),
        accumulator,
        mask=c_mask
    )

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper function to launch the Triton matmul kernel.
    """
    # Ensure tensors are on GPU and contiguous
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K_B, N = b.shape
    assert K == K_B, "Inner dimensions must match for matrix multiplication."

    # Prepare output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Get strides for the tensors
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Tunable block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid dimensions: one program per block of the output matrix C
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch the kernel
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
        Performs matrix multiplication of A and B using the Triton implementation.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        # Ensure inputs are moved to CUDA for Triton kernel execution
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
            
        return triton_matmul(A, B)