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
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offsets for M and N dimensions
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_m = tl.max_contiguous(tl.multiple_of(offsets_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offsets_n = tl.max_contiguous(tl.multiple_of(offsets_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # Create masks for bounds checking
    mask_m = offsets_m < M
    mask_n = offsets_n < N

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A slice: A.T is (M, K), so we need A[k, m] for A.T[m, k]
        # A_ptr has shape (K, M) with stride_am for M, stride_ak for K
        # For A.T[m, k] = A[k, m], we access A_ptr[k * stride_ak + m * stride_am]
        a_offsets = (
            offsets_k[:, None] * stride_ak +
            offsets_m[None, :] * stride_am
        )
        a_mask = (offsets_k[:, None] < K) & (offsets_m[None, :] < M)
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)

        # Load B slice: B.T is (N, K), so we need B[k, n] for B.T[n, k]
        # B_ptr has shape (N, K) with stride_bk for K, stride_bn for N
        # For B.T[n, k] = B[k, n], we access B_ptr[k * stride_bk + n * stride_bn]
        b_offsets = (
            offsets_k[:, None] * stride_bk +
            offsets_n[None, :] * stride_bn
        )
        b_mask = (offsets_k[:, None] < K) & (offsets_n[None, :] < N)
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)

        # Accumulate the product
        accumulator += tl.dot(a, b)

    # Store result
    c_offsets = (
        offsets_m[:, None] * stride_cm +
        offsets_n[None, :] * stride_cn
    )
    c_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    
    # Convert to output type
    c = accumulator.to(C_ptr.dtype.element_ty)
    tl.store(C_ptr + c_offsets, c, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication equivalent to torch.matmul(A.T, B.T)
    
    A: (K, M) -> A.T: (M, K)
    B: (N, K) -> B.T: (N, K)
    Result: (M, K) @ (N, K).T = (M, K) @ (K, N) = (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # A is (K, M), B is (N, K)
    # We want A.T @ B.T = (M, K) @ (K, N) = (M, N)
    K, M = A.shape
    N, _ = B.shape
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Set up strides
    stride_am = A.stride(1)
    stride_ak = A.stride(0)
    stride_bk = B.stride(1)
    stride_bn = B.stride(0)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    
    # Set block sizes for optimization
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    grid = (num_pid_m * num_pid_n,)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernel.
    Equivalent to torch.matmul(A.T, B.T) but optimized with custom kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using optimized Triton kernel.
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)