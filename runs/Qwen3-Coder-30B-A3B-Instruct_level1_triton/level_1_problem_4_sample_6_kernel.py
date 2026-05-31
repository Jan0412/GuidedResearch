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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map program ID to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator with zeros.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension.
    for k in range(0, K, BLOCK_SIZE_K):
        # Load a and b.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        # Compute accumulator.
        accumulator += tl.dot(a, b)
        # Advance pointers.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation if specified.
    if ACTIVATION == "relu":
        accumulator = tl.where(accumulator > 0, accumulator, 0.0)
        
    # Write back result.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Performs matrix multiplication using a Triton kernel.
    """
    assert a.is_cuda and b.is_cuda, "Both tensors must be on CUDA."
    assert a.dim() == 2 and b.dim() == 2, "Both tensors must be 2D."
    assert a.shape[1] == b.shape[0], f"Matrix shapes {a.shape} and {b.shape} are incompatible for multiplication."

    # Ensure inputs are contiguous
    a = a.contiguous()
    b = b.contiguous()

    # Prepare output tensor
    m, k = a.shape
    k_, n = b.shape
    assert k == k_, f"Inner dimensions mismatch: {k} vs {k_}"

    c = torch.empty((m, n), device=a.device, dtype=torch.float32)

    # Define launch parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Determine the number of program IDs needed
    num_warps = 4
    num_programs_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(n, BLOCK_SIZE_N)
    num_programs = num_programs_m * num_programs_n
    
    # Configure grid
    grid = (num_programs, 1, 1)
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        GROUP_SIZE_M,
        'none'
    )
    
    return c

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix-vector multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix-vector multiplication using Triton kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matmul(A, B)

M = 256 * 8 # 2048
K = 131072 * 8 # 1048576

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]

def get_init_inputs():
    return []