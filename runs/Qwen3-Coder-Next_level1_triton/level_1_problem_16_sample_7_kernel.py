import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Triton kernel for optimized matrix multiplication with transpose
@triton.jit
def matmul_transpose_kernel(
    A_ptr,  # Pointer to matrix A (shape: K, M) - we'll use A.T internally
    B_ptr,  # Pointer to matrix B (shape: K, N)
    C_ptr,  # Pointer to output matrix C (shape: M, N)
    M, N, K,  # Dimensions
    stride_am, stride_ak,  # Strides for A (transposed view: A_T[m, k] = A[k, m])
    stride_bk, stride_bn,  # Strides for B
    stride_cm, stride_cn,  # Strides for C
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create tile offsets for M and N dimensions
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Grouping logic to improve cache reuse
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    pid_n = tl.program_id(1)  # Recalculate in case of grouping

    # Create block offsets for M and N
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create block offsets for K dimension
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointers to the blocks of A and B
    # Note: A is accessed as A.T, so we use A_ptr + k * stride_ak + m * stride_am
    # which is equivalent to accessing A_T[m, k] = A[k, m]
    a_ptrs = A_ptr + (offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator for the matrix multiplication
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load a block of A (transposed) and B
        # Use masking to handle cases where K is not divisible by BLOCK_SIZE_K
        k_mask = k * BLOCK_SIZE_K + offs_k < K
        a = tl.load(a_ptrs, mask=k_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        
        # Accumulate the product (note: we're doing A.T @ B, so it's A[k,m] * B[k,n])
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Convert accumulator to float16 if needed and store result
    c = accumulator.to(tl.float32)  # Keep as float32 for precision
    
    # Write back to output
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul_transpose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A.T @ B using custom Triton kernel
    
    Args:
        A: Input tensor of shape (K, M)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    K, M = A.shape
    K2, N = B.shape
    assert K == K2, "Dimension mismatch: A.shape[0] must equal B.shape[0]"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Define grid function
    grid = (num_pid_m, num_pid_n)
    
    # Launch kernel
    matmul_transpose_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model that uses Triton kernel for matrix multiplication.
    Performs C = A.T @ B using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A.T @ B using optimized Triton kernel.

        Args:
            A: Input tensor of shape (M, K) in original, but we expect (K, M) in our kernel.
               We'll transpose if needed to match expected input format.
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Note: The original model takes A of shape (M, K) and B of shape (K, N)
        # and computes A.T @ B, which is (K, M) @ (K, N) -> (M, N)
        # Our Triton kernel expects A of shape (K, M) and B of shape (K, N)
        # So if A is passed as (M, K), we need to transpose it to (K, M) first
        if A.shape[0] != B.shape[0]:  # If A is (M, K) and B is (K, N), we need to handle this
            # Transpose A to get (K, M)
            A_t = A.T.contiguous()
            return triton_matmul_transpose(A_t, B)
        else:
            # If A is already (K, M), use it directly
            return triton_matmul_transpose(A, B)