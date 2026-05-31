import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_4d_2d_kernel(
    A_ptr,  # 4D tensor pointer (b, i, j, l)
    B_ptr,  # 2D matrix pointer (l, k)
    C_ptr,  # Output 4D tensor pointer (b, i, j, k)
    B_stride_l, B_stride_k,  # Strides for B tensor
    C_stride_b, C_stride_i, C_stride_j, C_stride_k,  # Strides for C tensor
    B_size_l, B_size_k,  # Dimensions of B
    B_size_i, B_size_j,  # Dimensions of A (excluding l)
    total_elements,  # Total number of elements in output C
    BLOCK_SIZE_M: tl.constexpr,  # Block size for M dimension (k)
    BLOCK_SIZE_N: tl.constexpr,  # Block size for N dimension (l)
    BLOCK_SIZE_K: tl.constexpr,  # Block size for reduction dimension (l)
):
    # Calculate global indices
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(B_size_k, BLOCK_SIZE_M)
    num_pid_n = B_size_i * B_size_j
    pid_n = pid // num_pid_m
    pid_m = pid % num_pid_m
    
    # Map pid_n to (i_idx, j_idx)
    i_idx = pid_n // B_size_j
    j_idx = pid_n % B_size_j
    
    # Offsets for M dimension (k)
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Offsets for N dimension (l)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for M dimension
    mask_m = m_offsets < B_size_k
    
    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over K dimension (l)
    for k_start in range(0, B_size_l, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < B_size_l
        
        # Load A: shape (i_idx, j_idx, k_offsets) -> A[b=0, i_idx, j_idx, k_offsets]
        # We need to get A[0, i_idx, j_idx, k_offsets] but we can use stride info
        a_ptrs = A_ptr + i_idx * (B_size_j * B_size_l) + j_idx * B_size_l + k_offsets
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        
        # Load B: shape (k_offsets, m_offsets) -> B[k_offsets, m_offsets]
        b_ptrs = B_ptr + k_offsets[:, None] * B_stride_l + m_offsets[None, :] * B_stride_k
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
        
        # Accumulate product: a * b
        # a has shape (BLOCK_SIZE_K,), b has shape (BLOCK_SIZE_K, BLOCK_SIZE_M)
        acc += tl.sum(a[:, None] * b, axis=0)
    
    # Convert to output type and store
    c = acc.to(tl.float32)
    c_ptrs = C_ptr + i_idx * (B_size_j * B_size_k * C_stride_b) + j_idx * (B_size_k * C_stride_b) + m_offsets * C_stride_b
    tl.store(c_ptrs, c, mask=mask_m)


def triton_matmul_4d_2d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    
    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)
    
    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported."
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    b_size, i_size, j_size, l_size = A.shape
    _, k_size = B.shape
    
    # Create output tensor
    C = torch.empty((b_size, i_size, j_size, k_size), dtype=torch.float32, device=A.device)
    
    # Configure kernel parameters
    # Use block sizes that work well for GPUs (multiple of 16 or 32)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    num_pid_m = (k_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = i_size * j_size
    grid = (num_pid_m * num_pid_n,)
    
    # Launch kernel
    matmul_4d_2d_kernel[grid](
        A, B, C,
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        l_size, k_size,
        i_size, j_size,
        C.numel(),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of the model using Triton kernel for 4D tensor-matrix multiplication.
    Performs C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return triton_matmul_4d_2d(A, B)