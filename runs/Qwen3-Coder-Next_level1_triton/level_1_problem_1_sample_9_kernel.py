import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create block offsets
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    amask = offset_m < N
    anmask = offset_m < N
    bnmask = offset_n < N
    bknmask = offset_k < N
    cmask = offset_m < N
    cnmask = offset_n < N
    cmnmask = (offset_m[:, None] < N) & (offset_n[None, :] < N)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_SIZE_K):
        # Load A block
        a_offset = (offset_m[:, None] * stride_am + 
                   (k + offset_k[None, :]) * stride_ak)
        a = tl.load(A_ptr + a_offset, 
                   mask=(anmask[:, None] & bknmask[None, :]), 
                   other=0.0)
        
        # Load B block
        b_offset = ((k + offset_k[:, None]) * stride_bk + 
                   offset_n[None, :] * stride_bn)
        b = tl.load(B_ptr + b_offset, 
                   mask=(bknmask[:, None] & bnmask[None, :]), 
                   other=0.0)
        
        # Accumulate matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
    
    # Cast accumulator to float16 if needed (but keeping as float32 for FP32 precision)
    accumulator = accumulator.to(tl.float32)
    
    # Store result
    c_offset = (offset_m[:, None] * stride_cm + 
               offset_n[None, :] * stride_cn)
    tl.store(C_ptr + c_offset, accumulator, mask=cmnmask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication C = A @ B using Triton kernel.
    Optimized for square matrices with FP32 precision.
    """
    # Ensure inputs are contiguous and on GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Check dimensions
    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert B.shape[0] == B.shape[1], "Matrix B must be square"
    assert A.shape[0] == B.shape[0], "Matrices must have same dimension for multiplication"
    
    N = A.shape[0]
    
    # Create output tensor
    C = torch.empty((N, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes for optimization
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    num_pid_m = triton.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    grid = (num_pid_m * num_pid_n,)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom Triton kernel instead of PyTorch's built-in matmul.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return triton_matmul(A, B)