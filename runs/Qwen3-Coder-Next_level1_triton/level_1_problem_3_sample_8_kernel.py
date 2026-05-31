import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batched_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, m, n, k,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Get batch ID
    batch_id = tl.program_id(2)
    
    # Compute matrix C's block coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Number of program blocks along M dimension
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    
    # Grouping for better cache utilization
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    pid_n = tl.program_id(1)
    
    # Create offsets for M and N blocks
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create pointers for A and B
    # A: [batch_id, m_block, k]
    # B: [batch_id, k, n_block]
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr + batch_id * stride_ab,
        shape=(m, k),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr + batch_id * stride_bb,
        shape=(k, n),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1)
    )
    
    # Create pointer for C output
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr + batch_id * stride_cb,
        shape=(m, n),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k_offset in range(0, k, BLOCK_SIZE_K):
        # Load A and B blocks
        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Matrix multiply
        acc += tl.dot(a, b, out_dtype=tl.float32)
        
        # Advance pointers
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_SIZE_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_SIZE_K, 0))
    
    # Store result
    tl.store(C_block_ptr, acc.to(C_ptr.type.element_ty), boundary_check=(0, 1))


def triton_batched_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized batched matrix multiplication using Triton.
    
    Args:
        A: Input tensor of shape (batch_size, m, k)
        B: Input tensor of shape (batch_size, k, n)
    
    Returns:
        C: Output tensor of shape (batch_size, m, n)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 3 and B.dim() == 3, "Inputs must be 3D tensors"
    assert A.shape[0] == B.shape[0], "Batch dimensions must match"
    assert A.shape[2] == B.shape[1], "Inner dimensions must match for matrix multiplication"
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Create output tensor
    C = torch.empty((batch_size, m, n), device=A.device, dtype=A.dtype)
    
    # Configure block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Compute grid dimensions
    num_pid_m = triton.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(n, BLOCK_SIZE_N)
    
    grid = (num_pid_m, num_pid_n, batch_size)
    
    # Launch kernel
    batched_gemm_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for batched matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using optimized Triton kernel.
        
        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).
        
        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_batched_gemm(A, B)