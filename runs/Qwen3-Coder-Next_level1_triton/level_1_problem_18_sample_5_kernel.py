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
    # Matrix multiplication: C = A @ B
    # A shape: (M, K), B shape: (K, N), C shape: (M, N)
    
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create start offsets for M and N blocks
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create pointers for the blocks of A and B
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B blocks (with boundary checks)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate the matrix multiplication
        acc = tl.dot(a, b, acc)
        
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Cast to output dtype and store result
    c = acc.to(tl.float32)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton-based matrix multiplication: C = A @ B
    For the original problem, we want torch.matmul(A.T, B.T) = (B @ A).T
    So we compute B @ A and then transpose the result.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Original: torch.matmul(A.T, B.T)
    # This equals (B @ A).T
    # So we compute B @ A first, then transpose
    
    # B shape: (N, K), A shape: (K, M) -> result should be (N, M)
    # But the original operation gives (M, N), so we need to be careful
    
    # Let's trace through dimensions:
    # A.T has shape (M, K), B.T has shape (N, K)
    # torch.matmul(A.T, B.T) with shapes (M, K) @ (K, N) gives (M, N)
    # But wait - torch.matmul(A.T, B.T) where A.T is (M,K) and B.T is (N,K) 
    # doesn't work directly because (M,K) @ (N,K) is invalid (K != N)
    
    # Re-reading the original: torch.matmul(A.T, B.T)
    # A has shape (K, M) -> A.T has shape (M, K)
    # B has shape (N, K) -> B.T has shape (K, N)
    # So torch.matmul(A.T, B.T) = torch.matmul((M,K), (K,N)) = (M,N) ✓
    
    # This is equivalent to (B @ A).T because:
    # (B @ A).T = A.T @ B.T where A is (K,M), B is (N,K)
    # A.T is (M,K), B.T is (K,N) ✓
    
    # So we can compute B @ A which is (N,K) @ (K,M) = (N,M)
    # Then transpose to get (M,N)
    
    M, K1 = A.shape  # A: (K, M) in original - wait, let me check get_inputs()
    # get_inputs() says A = torch.rand(K, M) so A is (K, M)
    # B = torch.rand(N, K) so B is (N, K)
    
    # So A.T is (M, K), B.T is (K, N)
    # torch.matmul(A.T, B.T) = (M, K) @ (K, N) = (M, N)
    
    # We'll implement this directly as a matmul with proper strides
    
    # Extract actual dimensions
    K_in, M_in = A.shape  # A is (K, M)
    N_in, K2_in = B.shape  # B is (N, K)
    
    assert K_in == K2_in, f"Dimension mismatch: A has K={K_in}, B has K={K2_in}"
    
    K = K_in
    M = M_in
    N = N_in
    
    # Create output tensor of shape (M, N)
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure block sizes (tuned for FP32)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    # Grid dimensions
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
        GROUP_SIZE_M=8,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for matrix multiplication.
    Replaces torch.matmul(A.T, B.T) with an optimized custom implementation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)