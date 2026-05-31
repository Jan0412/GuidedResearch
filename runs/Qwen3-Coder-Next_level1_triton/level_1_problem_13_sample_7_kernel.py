import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,  # Dimension of the square matrices
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Matrix multiplication kernel for square matrices
    # Each program computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile of C
    
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create offset blocks for C
    offset_am = pid_m * BLOCK_SIZE_M
    offset_bn = pid_n * BLOCK_SIZE_N
    offset_k = 0
    
    # Create pointers for A and B
    a_ptrs = A_ptr + offset_am * stride_am + offset_k * stride_ak
    b_ptrs = B_ptr + offset_k * stride_bk + offset_bn * stride_bn
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_SIZE_K):
        # Load a block of A and B
        mask_a = (offset_am < N)[:, None] & ((k + tl.arange(0, BLOCK_SIZE_K)) < N)[None, :]
        mask_b = ((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < N) & (offset_bn < N)[None, :]
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Accumulate the block multiplication
        accumulator = tl.dot(a, b, accumulator)
        
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Store the result
    C_block = accumulator.to(tl.float32)
    mask_c = (offset_am < N)[:, None] & (offset_bn < N)[None, :]
    c_ptrs = C_ptr + offset_am * stride_cm + offset_bn * stride_cn
    tl.store(c_ptrs, C_block, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Matrix multiplication for square matrices using Triton kernel.
    Assumes A and B are both N x N symmetric matrices.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must have the same shape."
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1], "Matrices must be square."
    
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    
    # Create output tensor
    C = torch.empty_like(A)
    
    # Define block sizes for optimization
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    num_pid_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_pid_m * num_pid_n,)
    
    # Launch the kernel
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
    Optimized model that performs matrix multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return triton_matmul(A, B)