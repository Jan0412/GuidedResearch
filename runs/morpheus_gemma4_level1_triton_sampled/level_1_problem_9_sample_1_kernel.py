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
    # Map program IDs to the blocks of the output matrix C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers to the blocks of A and B
    # A is (M, K), B is (K, N)
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension (inner dimension)
    # Since K is small (e.g., 32), this loop may run only once if BLOCK_K >= K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load blocks of A and B
        # Shift pointers for the current K-block
        a = tl.load(a_ptrs + k * BLOCK_K * stride_ak, mask=(rm[:, None] < M) & (rk[None, :] + k * BLOCK_K < K), other=0.0)
        b = tl.load(b_ptrs + k * BLOCK_K * stride_bk, mask=(rk[:, None] + k * BLOCK_K < K) & (rn[None, :] < N), other=0.0)
        
        # Perform matrix multiplication block
        accumulator += tl.dot(a, b)

    # Store the result block back to C
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N))

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel.
    Optimized for cases where one dimension is significantly smaller than others.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Inner dimensions must match"
    
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Tuning parameters
    # Since N or K might be small, we choose blocks that fit well
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32 # Set to 32 as it's a common power of 2 and fits the target architecture
    
    # Grid calculation
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return c

class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using the optimized Triton kernel.

        Args:
            A (torch.Tensor): Input matrix.
            B (torch.Tensor): Input matrix.

        Returns:
            torch.Tensor: Output matrix.
        """
        return triton_matmul(A, B)