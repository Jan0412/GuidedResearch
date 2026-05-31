import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr,  # Pointer to input A with shape (batch_size, m, k)
    B_ptr,  # Pointer to input B with shape (batch_size, k, n)
    C_ptr,  # Pointer to output C with shape (batch_size, m, n)
    batch_size, m, n, k,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  # Tile size for M dimension
    BLOCK_SIZE_N: tl.constexpr,  # Tile size for N dimension
    BLOCK_SIZE_K: tl.constexpr,  # Tile size for K dimension
    GROUP_SIZE_M: tl.constexpr,  # Group size for better load balancing
):
    # Get batch index
    batch_idx = tl.program_id(2)
    pid = tl.program_id(0)
    
    # Create a grid for M and N dimensions
    num_pid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create offsets for M and N dimensions
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    am_mask = offsets_am < m
    bn_mask = offsets_bn < n
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k_offset in range(0, k, BLOCK_SIZE_K):
        k_offsets = k_offset + offsets_k
        k_mask = k_offsets < k
        
        # Load tile from A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = (
            batch_idx * stride_ab +
            offsets_am[:, None] * stride_am +
            k_offsets[None, :] * stride_ak
        )
        a = tl.load(A_ptr + a_offsets, mask=k_mask[None, :], other=0.0)
        
        # Load tile from B: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_offsets = (
            batch_idx * stride_bb +
            k_offsets[:, None] * stride_bk +
            offsets_bn[None, :] * stride_bn
        )
        b = tl.load(B_ptr + b_offsets, mask=k_mask[:, None], other=0.0)
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Convert accumulator to float16 if needed (but keeping float32 for FP32 precision)
    accumulator = accumulator.to(tl.float32)
    
    # Store result
    c_offsets = (
        batch_idx * stride_cb +
        offsets_am[:, None] * stride_cm +
        offsets_bn[None, :] * stride_cn
    )
    c_mask = (am_mask[:, None] & bn_mask[None, :])
    tl.store(C_ptr + c_offsets, accumulator, mask=c_mask)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs batched matrix multiplication using Triton kernel.
    
    Args:
        A: Input tensor of shape (batch_size, m, k)
        B: Input tensor of shape (batch_size, k, n)
    
    Returns:
        C: Output tensor of shape (batch_size, m, n)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    # Ensure inputs are float32
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    
    # Create output tensor
    C = torch.empty((batch_size, m, n), dtype=torch.float32, device=A.device)
    
    # Get strides
    stride_ab, stride_am, stride_ak = A.stride()
    stride_bb, stride_bk, stride_bn = B.stride()
    stride_cb, stride_cm, stride_cn = C.stride()
    
    # Define block sizes - tuned for FP32 performance
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    num_pid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_programs = num_pid_m * num_pid_n * batch_size
    
    # Define grid
    grid = lambda meta: (
        min(65535, num_pid_m * num_pid_n),
        1,
        batch_size
    )
    
    # Launch kernel
    bmm_kernel[grid](
        A, B, C,
        batch_size, m, n, k,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    Optimized with Triton kernel for improved performance.
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
        return triton_bmm(A, B)