import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
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
    
    # Block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets for M and N dimensions
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Masks
    masks_m = offsets_m < M
    masks_n = offsets_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute actual K offsets
        k_offset = k + offsets_k
        masks_k = k_offset < K
        
        # Load A block: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_ptrs = A + (offsets_m[:, None] * stride_am + k_offset[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=masks_m[:, None] & masks_k[None, :], other=0.0)
        
        # Load B block: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_ptrs = B + (k_offset[:, None] * stride_bk + offsets_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=masks_k[:, None] & masks_n[None, :], other=0.0)
        
        # Matrix multiply
        accumulator += tl.dot(a, b)
    
    # Convert accumulator to output type and store
    C_block = accumulator.to(C.dtype.element_ty)
    
    # Store result
    c_ptrs = C + (offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn)
    tl.store(c_ptrs, C_block, mask=masks_m[:, None] & masks_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A * B using Triton kernel.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (K, N)
        
    Returns:
        Output tensor of shape (M, N)
    """
    # Ensure tensors are on GPU and contiguous
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Incompatible dimensions: A.shape[1] must equal B.shape[0]"
    
    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid dimensions
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A * B)
    using a custom Triton kernel instead of PyTorch's built-in matmul.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)