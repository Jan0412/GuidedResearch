import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Map program ID to the block of C it computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for the current block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks in A and B
    # A is (M, K), B is (N, K). We compute A @ B.T
    # A block: [BLOCK_SIZE_M, BLOCK_SIZE_K]
    # B block: [BLOCK_SIZE_K, BLOCK_SIZE_N] -> loaded from B (N, K)
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    # To get B.T block [BLOCK_SIZE_K, BLOCK_SIZE_N], we load B[rn, rk] 
    # and treat the K dimension as the first dimension of the dot product.
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks with masking for boundaries
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < K - k * BLOCK_SIZE_K) & (rn[None, :] < N), other=0.0)
        
        # Perform matrix multiplication block
        accumulator += tl.dot(a, b)
        
        # Advance pointers to the next block along K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store the result in C
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N) )

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are contiguous and on GPU
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    N, K_b = b.shape
    assert K == K_b, "Inner dimensions must match"
    
    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Strides
    stride_am, stride_ak = a.stride()
    stride_bn, stride_bk = b.stride()
    stride_cm, stride_cn = c.stride()
    
    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Grid definition
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c

class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A * B^T)
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication A @ B.T.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)