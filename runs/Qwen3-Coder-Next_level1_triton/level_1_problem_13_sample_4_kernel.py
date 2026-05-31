import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    num_pid_in_group = GROUP_SIZE * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create block offsets
    off_m = pid_m * BLOCK_SIZE
    off_n = pid_n * BLOCK_SIZE
    
    # pointers to A and B
    A = A + off_m * stride_am + off_n * stride_ak  # This is wrong, fix it
    # Correct: A pointer should be off_m * stride_am + (off_k) * stride_ak
    # Let's fix the kernel with proper indexing
    
    # Reimplement with correct indexing
    pass


@triton.jit
def matmul_kernel_correct(
    A, B, C,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    num_pid_in_group = GROUP_SIZE * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create block offsets
    off_m = pid_m * BLOCK_SIZE
    off_n = pid_n * BLOCK_SIZE
    
    # Create block pointers for A and B
    # A: [M, K] -> [off_m, off_k]
    # B: [K, N] -> [off_k, off_n]
    # C: [M, N] -> [off_m, off_n]
    
    # Initialize block pointers
    A_block_ptr = tl.make_block_ptr(
        base=A,
        shape=(N, N),
        strides=(stride_am, stride_ak),
        offsets=(off_m, 0),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
        order=(1, 0)
    )
    B_block_ptr = tl.make_block_ptr(
        base=B,
        shape=(N, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, off_n),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
        order=(0, 1)
    )
    C_block_ptr = tl.make_block_ptr(
        base=C,
        shape=(N, N),
        strides=(stride_cm, stride_cn),
        offsets=(off_m, off_n),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
        order=(1, 0)
    )
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, N, BLOCK_SIZE):
        # Load blocks
        a = tl.load(A_block_ptr, boundary_check=(1,), padding_option='zero')
        b = tl.load(B_block_ptr, boundary_check=(0,), padding_option='zero')
        
        # Matrix multiply
        accumulator += tl.dot(a, b)
        
        # Update block pointers
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_SIZE))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_SIZE, 0))
    
    # Store result
    tl.store(C_block_ptr, accumulator.to(C.dtype.element_ty), boundary_check=(0, 1))


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Triton-based matrix multiplication optimized for symmetric matrices.
    
    Args:
        A: Input matrix A, shape (N, N)
        B: Input matrix B, shape (N, N)
    
    Returns:
        torch.Tensor: Output matrix C, shape (N, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    N = A.shape[0]
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = 128
    GROUP_SIZE = 8
    
    # Calculate grid
    num_pid_m = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pid_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_pid_m * num_pid_n * GROUP_SIZE,)
    
    # Launch kernel
    matmul_kernel_correct[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernels.
    Optimized for symmetric matrices with tiling and shared memory usage.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication using optimized Triton kernel.
        
        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.
        
        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return triton_matmul(A, B)