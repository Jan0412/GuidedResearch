import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_kernel(
    A_ptr,  # Pointer to A (K, M) - will be treated as A^T (M, K)
    B_ptr,  # Pointer to B (K, N)
    C_ptr,  # Pointer to output C (M, N)
    M, N, K,
    stride_am, stride_ak,  # Strides for A (K, M) layout: A[k, m] -> A_ptr[k * stride_ak + m * stride_am]
    stride_bk, stride_bn,  # Strides for B (K, N) layout: B[k, n] -> B_ptr[k * stride_bk + n * stride_bn]
    stride_cm, stride_cn,  # Strides for C (M, N) layout: C[m, n] -> C_ptr[m * stride_cm + n * stride_cn]
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_programs = num_programs_m * num_programs_n
    
    # Group programs for better cache locality (similar to CUTLASS)
    num_programs_in_group = GROUP_SIZE_M * num_programs_n
    group_id = pid // num_programs_in_group
    first_program_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_programs_m - first_program_m, GROUP_SIZE_M)
    pid_m = first_program_m + (pid % group_size_m)
    pid_n = (pid % num_programs_in_group) // group_size_m
    
    # Create offsets for the block of A^T (M, K) and B (K, N)
    # For A^T, row = pid_m, col = range of K values
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointers for the blocks
    # A is stored as (K, M), so A^T element [m, k] is A[k, m]
    # A_ptr[k * stride_ak + m * stride_am] where stride_ak=1 (contiguous in M), stride_am=K
    # So for A^T block [m_offset, k_offset], we need A_ptr[k_offset * 1 + m_offset * K]
    # But since we're iterating over K in blocks, we'll use A_ptr + offsets_k[:, None] * stride_ak + offsets_m[None, :] * stride_am
    A_ptr_block = A_ptr + offsets_k[:, None] * stride_ak + offsets_m[None, :] * stride_am
    
    # B is (K, N), so B_ptr[k * stride_bk + n * stride_bn] with stride_bk=1, stride_bn=N
    B_ptr_block = B_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    
    # Accumulator for the matrix multiplication
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_offset in range(0, K, BLOCK_SIZE_K):
        # Load blocks from A^T (M, K) and B (K, N)
        # Note: For A, we're loading from the actual A tensor but indexing as if it were A^T
        a_tile = tl.load(A_ptr_block, mask=offsets_k[:, None] < K - k_offset, other=0.0)
        b_tile = tl.load(B_ptr_block, mask=offsets_k[:, None] < K - k_offset, other=0.0)
        
        # Perform matrix multiplication: C += A^T @ B
        # a_tile has shape (BLOCK_SIZE_K, BLOCK_SIZE_M) which is (K_block, M_block)
        # b_tile has shape (BLOCK_SIZE_K, BLOCK_SIZE_N) which is (K_block, N_block)
        # We want A^T @ B where A^T is (M_block, K_block) and B is (K_block, N_block)
        # So we need to transpose a_tile to get (BLOCK_SIZE_M, BLOCK_SIZE_K)
        accumulator += tl.dot(a_tile, b_tile, trans_a=True, trans_b=False)
        
        # Update pointers for next iteration
        A_ptr_block += BLOCK_SIZE_K * stride_ak
        B_ptr_block += BLOCK_SIZE_K * stride_bk
    
    # Store the result
    # Only store valid elements within bounds
    C_ptr_block = C_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Convert to output type and store
    tl.store(C_ptr_block, accumulator.to(C_ptr.dtype.element_ty), mask=mask)


def triton_matmul_transpose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A^T @ B where A has shape (K, M) and B has shape (K, N)
    Output C has shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D tensors."
    assert A.shape[0] == B.shape[0], f"A and B must have same K dimension, got {A.shape[0]} and {B.shape[0]}"
    
    A = A.contiguous()
    B = B.contiguous()
    
    K, M = A.shape
    _, N = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure block sizes for optimal performance (FP32)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    num_programs_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_programs_m * num_programs_n,)
    
    # Launch kernel
    matmul_transpose_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),  # stride_ak, stride_am for A(K,M)
        B.stride(0), B.stride(1),  # stride_bk, stride_bn for B(K,N)
        C.stride(0), C.stride(1),  # stride_cm, stride_cn for C(M,N)
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication C = A^T * B using Triton kernel
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication using Triton kernel.
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).
        
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul_transpose(A, B)