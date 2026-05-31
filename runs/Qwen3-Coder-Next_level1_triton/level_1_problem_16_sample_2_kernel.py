import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_T_ptr,  # Pointer to transposed A (shape: K x M)
    B_ptr,    # Pointer to B (shape: K x N)
    C_ptr,    # Pointer to output C (shape: M x N)
    M, N, K,
    stride_A_t_m, stride_A_t_k,
    stride_B_k, stride_B_n,
    stride_C_m, stride_C_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Number of programs per row and column
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouping for better cache utilization
    num_programs_per_group = GROUP_SIZE_M * num_programs_n
    group_id = pid_m // GROUP_SIZE_M
    first_program_m = group_id * GROUP_SIZE_M
    rest_programs_m = pid_m % GROUP_SIZE_M
    
    # Adjust pid_m and pid_n for grouping
    pid_m = first_program_m + rest_programs_m
    pid_n = (pid_m % num_programs_n) if group_id == 0 else (pid_m // num_programs_n)
    
    # Create offset arrays for block indices
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    m_mask = offsets_m < M
    n_mask = offsets_n < N
    kn_mask = (offsets_k < K)[None, :] & (offsets_n < N)[None, :]
    mk_mask = (offsets_m < M)[:, None] & (offsets_k < K)[None, :]
    
    # Initialize accumulator for C matrix
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A^T (A_T is K x M)
        a_t_offsets = (offsets_k[:, None] * stride_A_t_k + offsets_m[None, :] * stride_A_t_m)
        a_t = tl.load(
            A_T_ptr + a_t_offsets,
            mask=(offsets_k[:, None] < K) & (offsets_m[None, :] < M),
            other=0.0
        )
        
        # Load blocks of B (K x N)
        b_offsets = (offsets_k[:, None] * stride_B_k + offsets_n[None, :] * stride_B_n)
        b = tl.load(
            B_ptr + b_offsets,
            mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N),
            other=0.0
        )
        
        # Perform matrix multiplication accumulation: C += A^T * B
        acc += tl.dot(a_t, b, out_dtype=tl.float32)
        
        # Update k offset for next iteration
        offsets_k += BLOCK_SIZE_K
    
    # Store result to C (M x N)
    c_offsets = (offsets_m[:, None] * stride_C_m + offsets_n[None, :] * stride_C_n)
    c_mask = (m_mask[:, None] & n_mask[None, :])
    
    # Convert to float32 if needed (for FP32 case, it's already float)
    tl.store(
        C_ptr + c_offsets,
        acc.to(tl.float32),
        mask=c_mask
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A.T @ B using a Triton kernel.
    
    Args:
        A: Input tensor of shape (K, M)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Ensure input dtypes are float32
    if A.dtype != torch.float32:
        A = A.to(torch.float32)
    if B.dtype != torch.float32:
        B = B.to(torch.float32)
    
    K, M = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match for matrix multiplication"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Define block sizes (tunable parameters for performance)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Define grid
    grid = (num_blocks_m, num_blocks_n)
    
    # Launch kernel
    matmul_kernel[grid](
        A_T_ptr=A,
        B_ptr=B,
        C_ptr=C,
        M=M,
        N=N,
        K=K,
        stride_A_t_m=A.stride(0),
        stride_A_t_k=A.stride(1),
        stride_B_k=B.stride(0),
        stride_B_n=B.stride(1),
        stride_C_m=C.stride(0),
        stride_C_n=C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using optimized Triton kernel.
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).
        
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)