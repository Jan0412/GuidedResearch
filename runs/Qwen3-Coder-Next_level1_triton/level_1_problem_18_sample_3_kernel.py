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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create tile offsets
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    pid_n = (pid_n % num_pid_n)
    
    # Create block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets for M and N dimensions
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_m = tl.multiple_of(offsets_m, BLOCK_SIZE_M)
    offsets_n = tl.multiple_of(offsets_n, BLOCK_SIZE_N)
    
    # Create mask for out-of-bounds
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k_offset in range(0, K, BLOCK_SIZE_K):
        offsets_k = k_offset + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A tile: A is (M, K), so we need (offsets_m, offsets_k)
        a_offsets = offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        a_mask = mask_m[:, None] & (offsets_k[None, :] < K)
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load B tile: B is (K, N), so we need (offsets_k, offsets_n)
        b_offsets = offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        b_mask = (offsets_k[:, None] < K) & mask_n[None, :]
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Matrix multiply
        accumulator += tl.dot(a, b)
    
    # Store result
    c_offsets = offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + c_offsets, accumulator, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B using Triton kernel.
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Inner dimensions must match"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Define block sizes (tunable parameters for performance)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
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
    Optimized model that performs matrix multiplication using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A.T @ B.T using Triton kernel.
        
        Args:
            A: Input tensor of shape (M, K) in the original architecture description.
            B: Input tensor of shape (K, N) in the original architecture description.
            
        Note: The actual inputs in get_inputs have shapes (K, M) and (N, K) respectively,
        so A.T has shape (M, K) and B.T has shape (K, N), which are compatible for matmul.
        
        Returns:
            Output tensor of shape (M, N).
        """
        # Transpose inputs to match expected shapes for the kernel
        A_T = A.T.contiguous()  # shape (M, K)
        B_T = B.T.contiguous()  # shape (K, N)
        
        # Use Triton kernel for matrix multiplication
        return triton_matmul(A_T, B_T)