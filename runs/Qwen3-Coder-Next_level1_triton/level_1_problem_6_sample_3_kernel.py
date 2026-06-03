import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.ops.matmul import matmul as triton_matmul


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride information
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Map program ids to the block of the matrix it should compute
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create offsets for blocks of the output matrix
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Ensure we don't go out of bounds
    off_m = tl.max_contiguous(tl.multiple_of(off_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    off_n = tl.max_contiguous(tl.multiple_of(off_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointers for the blocks of A and B
    A_ptrs = A_ptr + (off_m[:, None] * stride_am + off_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (off_k[:, None] * stride_bk + off_n[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the blocks of A and B
        a = tl.load(A_ptrs, mask=off_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(B_ptrs, mask=off_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Accumulate the dot product
        acc += tl.dot(a, b, out_dtype=tl.float32)
        
        # Update pointers for next iteration
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Convert to output dtype and store
    C_block = acc.to(tl.float32)
    
    # Store the block of the output matrix
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_m = tl.max_contiguous(tl.multiple_of(off_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    off_n = tl.max_contiguous(tl.multiple_of(off_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    
    C_ptrs = C_ptr + (off_m[:, None] * stride_cm + off_n[None, :] * stride_cn)
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    tl.store(C_ptrs, C_block, mask=mask)


def triton_matmul_custom(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton matrix multiplication kernel for FP32 tensors.
    Optimized for large K dimension (like in the given architecture).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, f"Matrix multiplication not possible: A has shape {A.shape}, B has shape {B.shape}"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # Define block sizes optimized for large K (131072 * 4 = 524,288)
    # These values are tuned for the given M=256, N=256, K=524,288
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 128
    GROUP_SIZE_M = 8
    
    # Grid definition
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using custom Triton kernel.
    Optimized for large K dimension (M=256, N=256, K=524,288).
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using custom Triton kernel.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        return triton_matmul_custom(A, B)