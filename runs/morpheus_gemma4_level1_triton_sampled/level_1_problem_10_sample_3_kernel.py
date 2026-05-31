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

    # Pointers to the start of the blocks in A and B
    # A has shape (M, K), B has shape (K, N)
    a_ptr += (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr += (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks from A and B with boundary masking
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (k * BLOCK_SIZE_K + rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr, mask=(k * BLOCK_SIZE_K + rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        
        # Matrix multiplication of blocks
        accumulator += tl.dot(a, b)
        
        # Advance pointers to the next block along the K dimension
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # Store the result in C
    c_ptr += (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr, accumulator, mask=c_mask)


def triton_matmul_3d(A, B):
    """
    Triton wrapper for 3D tensor-matrix multiplication.
    A: (N, M, K)
    B: (K, L)
    Output: (N, M, L)
    """
    # Flatten A to (N*M, K) and treat as 2D matmul
    N_batch, M_dim, K_dim = A.shape
    L_dim = B.shape[1]
    
    A_flat = A.view(-1, K_dim).contiguous()
    B_flat = B.contiguous()
    
    M_total = N_batch * M_dim
    N_total = L_dim
    K_total = K_dim

    # Output tensor
    C_flat = torch.empty((M_total, N_total), device=A.device, dtype=torch.float32)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid of blocks
    grid = (triton.cdiv(M_total, BLOCK_SIZE_M), triton.cdiv(N_total, BLOCK_SIZE_N))

    # Launch kernel
    matmul_kernel[grid](
        A_flat, B_flat, C_flat,
        M_total, N_total, K_total,
        A_flat.stride(0), A_flat.stride(1),
        B_flat.stride(0), B_flat.stride(1),
        C_flat.stride(0), C_flat.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Reshape result back to (N, M, L)
    return C_flat.view(N_batch, M_dim, L_dim)


class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L).
        """
        # Ensure inputs are on CUDA and in FP32
        A = A.cuda().float()
        B = B.cuda().float()
        return triton_matmul_3d(A, B)