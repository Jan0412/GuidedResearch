import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Grid coordinates
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)

    # Create offsets for the current block
    m_offsets = m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)

    # Create index matrices for A and B
    # A is (M, K), B is (K, N)
    # We need A[m, k] and B[k, n]
    a_offsets = m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
    b_offsets = k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    num_k_blocks = K // BLOCK_SIZE_K
    for k in range(num_k_blocks):
        # Load tiles from global memory
        a = tl.load(A_ptr + a_offsets + k * BLOCK_SIZE_K * stride_ak, mask=True)
        b = tl.load(B_ptr + b_offsets + k * BLOCK_SIZE_K * stride_bk, mask=True)

        # Perform matrix multiplication and accumulate
        # tl.dot is efficient and uses Tensor Cores when available
        acc = tl.dot(a, b, acc)

        # Update offsets for next block
        a_offsets += BLOCK_SIZE_K * stride_ak
        b_offsets += BLOCK_SIZE_K * stride_bk

    # Store result to global memory
    c_offsets = m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptr + c_offsets, acc, mask=True)


def triton_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for the Triton GEMM kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "FP32 precision required."
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Incompatible dimensions for matrix multiplication."
    
    # Output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Block sizes (tuned for typical performance)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    # Grid configuration
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication using a custom Triton GEMM kernel.
        
        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)
            
        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        # Reshape A to 2D: (b*i*j, l)
        b, i, j, l = A.shape
        A_2d = A.view(b * i * j, l)
        
        # B is already (l, k)
        
        # Perform GEMM using Triton kernel
        C_2d = triton_gemm(A_2d, B)
        
        # Reshape result back to 4D: (b, i, j, k)
        C = C_2d.view(b, i, j, -1)
        
        return C