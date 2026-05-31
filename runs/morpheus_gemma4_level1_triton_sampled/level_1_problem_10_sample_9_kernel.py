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
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication.
    C = A @ B
    A: (M, K), B: (K, N), C: (M, N)
    """
    # Map program ID to the tile of C it computes
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create offsets for the tiles
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the tiles
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles with masking for boundary conditions
        a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k * BLOCK_SIZE_K), other=0.0)
        
        # Matrix multiplication of tiles
        accumulator += tl.dot(a, b)
        
        # Advance pointers to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store the result in C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wrapper for the Triton matmul kernel that handles 3D tensor A.
    A: (N, M, K) -> Reshaped to (N*M, K)
    B: (K, L)
    Result: (N, M, L)
    """
    # Original dimensions
    N_batch, M_dim, K_dim = A.shape
    K_dim_B, L_dim = B.shape
    assert K_dim == K_dim_B, "Incompatible dimensions for matmul"

    # Flatten A to 2D for the kernel
    A_flat = A.view(-1, K_dim).contiguous()
    B = B.contiguous()
    
    M = A_flat.shape[0]  # N * M
    N = B.shape[1]       # L
    K = A_flat.shape[1]  # K

    # Output tensor
    C_flat = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Grid definition
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_kernel[grid](
        A_flat, B, C_flat,
        M, N, K,
        A_flat.stride(0), A_flat.stride(1),
        B.stride(0), B.stride(1),
        C_flat.stride(0), C_flat.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    # Reshape back to (N, M, L)
    return C_flat.view(N_batch, M_dim, L_dim)


class ModelNew(nn.Module):
    """
    Optimized 3D tensor-matrix multiplication using a custom Triton kernel.
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
        # Ensure inputs are on GPU and in FP32
        A = A.cuda().float()
        B = B.cuda().float()
        return triton_matmul(A, B)