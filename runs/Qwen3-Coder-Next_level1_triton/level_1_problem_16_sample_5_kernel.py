import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_a_kernel(
    A_ptr,  # Pointer to A with shape (K, M)
    B_ptr,  # Pointer to B with shape (K, N)
    C_ptr,  # Pointer to output C with shape (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Use GROUP_SIZE_M to improve cache performance
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    pid_n = pid_n
    
    # Create block pointers
    # For A^T: element (i,j) in A^T is element (j,i) in A
    # So we need to access A[j, i] for C[i, j]
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    
    # A^T is (M, K), so we need rows rm (M-dimension) and columns rk (K-dimension)
    # For A^T[r, c] = A[c, r], so we load A[rk, rm] where rk is from K, rm is from M
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(K, M),
        strides=(stride_ak, stride_am),
        offsets=(0, pid_m * BLOCK_SIZE_M),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M),
        order=(1, 0)
    )
    
    # B is (K, N), so we load B[rk, rn] where rk is from K, rn is from N
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles from A^T and B
        # A^T is (M, K), so tile is (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # But we're using block pointers, so we load from A directly
        a = tl.load(A_block_ptr, boundary_check=(1,), padding_option="zero")
        b = tl.load(B_block_ptr, boundary_check=(1,), padding_option="zero")
        
        # For A^T @ B: we need to compute sum over k of A^T[m,k] * B[k,n]
        # Since A^T[m,k] = A[k,m], this is sum over k of A[k,m] * B[k,n]
        # This is equivalent to matmul(A.T, B)
        
        # Transpose A tile to get A^T tile: a is (BLOCK_SIZE_K, BLOCK_SIZE_M), need (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_t = tl.trans(a)
        
        # Compute partial matrix multiplication
        acc += tl.dot(a_t, b)
        
        # Advance pointers
        A_block_ptr = tl.advance(A_block_ptr, (BLOCK_SIZE_K, 0))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_SIZE_K, 0))
    
    # Convert to output type and store
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    tl.store(C_block_ptr, acc.to(C_ptr.dtype.element_ty), boundary_check=(0, 1))


def triton_matmul_transpose_a(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A^T @ B using Triton kernel.
    
    Args:
        A: Input tensor of shape (K, M)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported"
    
    A = A.contiguous()
    B = B.contiguous()
    
    K, M = A.shape
    _, N = B.shape
    
    # Allocate output
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Configure block sizes (tuned for FP32)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    grid = (num_pid_m, num_pid_n)
    
    # Launch kernel
    matmul_transpose_a_kernel[grid](
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
    Optimized model that performs matrix multiplication C = A^T * B using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A^T @ B using optimized Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul_transpose_a(A, B)