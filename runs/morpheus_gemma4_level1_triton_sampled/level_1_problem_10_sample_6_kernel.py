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
    # Map program IDs to the block of the output matrix C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the first block of A and B
    # a_ptr is (M, K), b_ptr is (K, N)
    a_ptr += (pid_m * BLOCK_SIZE_M * stride_am)
    b_ptr += (pid_n * BLOCK_SIZE_N * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        # a_offsets: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = rm[:, None] * stride_am + (k * BLOCK_SIZE_K + rk)[None, :] * stride_ak
        # b_offsets: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_offsets = (k * BLOCK_SIZE_K + rk)[:, None] * stride_bk + rn[None, :] * stride_bn
        
        # Masking for boundaries
        a_mask = (rm[:, None] < M) & ((k * BLOCK_SIZE_K + rk)[None, :] < K)
        b_mask = ((k * BLOCK_SIZE_K + rk)[:, None] < K) & (rn[None, :] < N)
        
        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Matrix multiplication: C = A @ B
        accumulator += tl.dot(a, b)

    # Store the result in C
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)

def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    # A: (N, M, K), B: (K, L)
    # We treat A as (N*M, K) to perform a single large GEMM
    N_batch, M, K = A.shape
    K_B, L = B.shape
    assert K == K_B, "K dimensions must match"

    # Flatten A to (N*M, K)
    A_flat = A.contiguous().view(-1, K)
    M_total = A_flat.shape[0]
    
    B = B.contiguous()
    
    # Output tensor
    C_flat = torch.empty((M_total, L), device=A.device, dtype=torch.float32)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid dimensions
    grid = (triton.cdiv(M_total, BLOCK_SIZE_M), triton.cdiv(L, BLOCK_SIZE_N))

    # Strides
    stride_am, stride_ak = A_flat.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C_flat.stride()

    matmul_kernel[grid](
        A_flat, B, C_flat,
        M_total, L, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Reshape back to (N, M, L)
    return C_flat.view(N_batch, M, L)

class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L).
        """
        # Ensure inputs are on GPU and FP32
        A = A.cuda().float()
        B = B.cuda().float()
        return triton_matmul(A, B)