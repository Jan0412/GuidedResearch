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
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offset blocks for M and N dimensions
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for M and N dimensions
    m_mask = offsets_m < M
    n_mask = offsets_n < N
    # No mask needed for K dimension since we iterate through it completely
    
    # Create pointers for this tile
    a_ptrs = A_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = B_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    
    # Accumulator for the matrix multiplication
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles from A and B
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Store result
    c_ptrs = C_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Custom Triton kernel for matrix multiplication optimized for tall-and-skinny matrices.
    
    Args:
        A: Input matrix of shape (M, K) or (K, M)
        B: Input matrix of shape (K, N) or (N, K)
        
    Returns:
        Output matrix of shape (M, N) or (N, M)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Determine matrix dimensions
    if A.dim() == 2 and B.dim() == 2:
        # Handle standard 2D matrix multiplication
        if A.shape[1] == B.shape[0]:  # A is (M, K), B is (K, N)
            M, K = A.shape
            K_b, N = B.shape
            assert K == K_b, f"Matrix multiplication dimension mismatch: A.shape={A.shape}, B.shape={B.shape}"
            is_transposed = False
        elif A.shape[0] == B.shape[1]:  # A is (K, M), B is (N, K)
            K, M = A.shape
            N, K_b = B.shape
            assert K == K_b, f"Matrix multiplication dimension mismatch: A.shape={A.shape}, B.shape={B.shape}"
            is_transposed = True
            # Transpose inputs to standard form for kernel
            A = A.T.contiguous()
            B = B.T.contiguous()
            M, K = A.shape
            K_b, N = B.shape
        else:
            raise ValueError(f"Invalid matrix dimensions for multiplication: A.shape={A.shape}, B.shape={B.shape}")
    else:
        raise ValueError(f"Only 2D tensors supported, got A.dim()={A.dim()}, B.dim()={B.dim()}")
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set block sizes optimized for M >> N case
    BLOCK_SIZE_M = 128  # Larger block size for the M dimension (tall)
    BLOCK_SIZE_N = 32   # Smaller block size for the N dimension (skinny)
    BLOCK_SIZE_K = 32   # Reasonable block size for K dimension
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
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
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)