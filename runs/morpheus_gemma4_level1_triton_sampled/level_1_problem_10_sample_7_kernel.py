import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program IDs to the block of C it should compute
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # -----------------------------------------------------------
    # Create offsets for the blocks
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    rk = tl.arange(0, BLOCK_SIZE_K)

    # -----------------------------------------------------------
    # Pointers to the first blocks of A and B
    a_ptr += (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptr += (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the result matrix C
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the tiles
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        # Matrix multiply the tiles
        accumulator += tl.dot(a, b)

        # Advance the pointers to the next block
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Store the result in C
    c_ptr += (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr, accumulator, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    A: (N, M, K)
    B: (K, L)
    Returns: (N, M, L)
    """
    # Reshape A to (N*M, K) to treat it as a standard 2D matrix multiplication
    N_batch, M_dim, K_dim = A.shape
    L_dim = B.shape[1]
    
    A_flat = A.view(-1, K_dim).contiguous()
    B_flat = B.contiguous()
    
    M = A_flat.shape[0]
    N = B_flat.shape[1]
    K = B_flat.shape[0]

    C_flat = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid defines the number of blocks in M and N dimensions
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_kernel[grid](
        A_flat, B_flat, C_flat,
        M, N, K,
        A_flat.stride(0), A_flat.stride(1),
        B_flat.stride(0), B_flat.stride(1),
        C_flat.stride(0), C_flat.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return C_flat.view(N_batch, M_dim, L_dim)


class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication using custom Triton kernels.
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
        # Ensure tensors are on CUDA and in FP32
        A = A.cuda().float()
        B = B.cuda().float()
        return triton_matmul(A, B)