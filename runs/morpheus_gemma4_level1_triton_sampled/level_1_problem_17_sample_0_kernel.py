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
    # Map program IDs to the block of C it computes
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create offsets for the blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the blocks in A and B
    # A is (M, K), B is (N, K)
    # We compute C = A @ B.T
    a_ptr += pid_m * BLOCK_SIZE_M * stride_am
    b_ptr += pid_n * BLOCK_SIZE_N * stride_bn

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        # a_offsets shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = (rm[:, None] * stride_am + (k * BLOCK_SIZE_K + rk)[None, :] * stride_ak)
        # b_offsets shape: (BLOCK_SIZE_N, BLOCK_SIZE_K)
        b_offsets = (rn[:, None] * stride_bn + (k * BLOCK_SIZE_K + rk)[None, :] * stride_bk)
        
        a = tl.load(a_ptr + a_offsets, mask=(rm[:, None] < M) & ((k * BLOCK_SIZE_K + rk)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=(rn[:, None] < N) & ((k * BLOCK_SIZE_K + rk)[None, :] < K), other=0.0)
        
        # Compute dot product: (BLOCK_SIZE_M, BLOCK_SIZE_K) @ (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # Since b is (BLOCK_SIZE_N, BLOCK_SIZE_K), we transpose it
        accumulator += tl.dot(a, tl.trans(b))

    # Store the result in C
    c_offsets = (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptr + c_offsets, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N))

def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    # A: (M, K), B: (N, K) -> C: (M, N)
    M, K = A.shape
    N, _ = B.shape
    
    # Ensure tensors are contiguous and on GPU
    A = A.contiguous()
    B = B.contiguous()
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Strides
    stride_am, stride_ak = A.stride()
    stride_bn, stride_bk = B.stride()
    stride_cm, stride_cn = C.stride()

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid: one program for each block of C
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B^T) using a Triton kernel.
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