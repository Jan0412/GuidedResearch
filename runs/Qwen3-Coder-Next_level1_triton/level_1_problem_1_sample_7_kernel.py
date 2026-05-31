import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Matrix multiplication kernel using tiled approach with software pipelining
    
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Group ID and local IDs
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create block offsets
    offset_am = pid_m * BLOCK_SIZE_M
    offset_bn = pid_n * BLOCK_SIZE_N
    offset_k = 0
    
    # Allocate accumulator register
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tile from A matrix
        a_mask = (offset_am + tl.arange(0, BLOCK_SIZE_M))[:, None] < M
        a = tl.load(
            A_ptr + offset_am * stride_am + (offset_k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * stride_ak,
            mask=a_mask,
            other=0.0
        )
        
        # Load tile from B matrix
        b_mask = (offset_k + tl.arange(0, BLOCK_SIZE_K)[:, None]) < K
        b = tl.load(
            B_ptr + (offset_k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * stride_bk + offset_bn * stride_bn,
            mask=b_mask,
            other=0.0
        )
        
        # Accumulate matrix multiply
        accumulator += tl.dot(a, b)
        
        offset_k += BLOCK_SIZE_K
    
    # Convert accumulator to float16 if needed, but keep as float32 for precision
    c = accumulator.to(tl.float32)
    
    # Store result
    c_mask = ((offset_am + tl.arange(0, BLOCK_SIZE_M))[:, None] < M) & \
             ((offset_bn + tl.arange(0, BLOCK_SIZE_N)[None, :]) < N)
    tl.store(
        C_ptr + offset_am * stride_cm + offset_bn * stride_cn,
        c,
        mask=c_mask
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton-based matrix multiplication optimized for square matrices.
    Supports FP32 precision.
    """
    # Ensure inputs are contiguous and on GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Check dimensions
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 is supported"
    
    # Get dimensions
    M, K = A.shape
    _, N = B.shape
    
    # Allocate output
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Configure kernel launch parameters
    # Use heuristics for block sizes that work well for square matrices
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Compute grid dimensions
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_pid_m * num_pid_n,)
    
    # Calculate strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return triton_matmul(A, B)