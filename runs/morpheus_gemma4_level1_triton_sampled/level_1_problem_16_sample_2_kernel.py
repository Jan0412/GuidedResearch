import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_transpose_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel to compute C = A^T * B
    A is (K, M), B is (K, N), C is (M, N)
    A^T is (M, K)
    """
    # -----------------------------------------------------------
    # Map program ids to the block of C it should compute
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouping for L2 cache optimization
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) * GROUP_SIZE_M # This is a simplification; usually handled differently
    # Correct grouping logic:
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m)
    
    # To implement grouping properly:
    # pid = pid_m * num_pid_n + pid_n
    # But for simplicity and correctness in this specific case:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # -----------------------------------------------------------
    # Create offsets for the blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # -----------------------------------------------------------
    # Pointers to the first blocks of A and B
    # A is (K, M), we want A^T which is (M, K)
    # A^T[rm, rk] = A[rk, rm]
    # Offset for A[rk, rm] = rk * stride_ak + rm * stride_am
    a_ptr += (0 * stride_ak + 0 * stride_am)
    b_ptr += (0 * stride_bk + 0 * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the result matrix C
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load block from A^T (which is a block from A)
        # a_block shape: (BLOCK_SIZE_K, BLOCK_SIZE_M)
        a_offsets = (rk[:, None] * stride_ak + rm[None, :] * stride_am)
        a_block = tl.load(a_ptr + a_offsets, mask=(rk[:, None] + k * BLOCK_SIZE_K < K) & (rm[None, :] < M), other=0.0)
        
        # Load block from B
        # b_block shape: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_offsets = (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        b_block = tl.load(b_ptr + b_offsets, mask=(rk[:, None] + k * BLOCK_SIZE_K < K) & (rn[None, :] < N), other=0.0)
        
        # Matrix multiply: (BLOCK_SIZE_M, BLOCK_SIZE_K) @ (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # We transpose a_block to get (BLOCK_SIZE_M, BLOCK_SIZE_K)
        accumulator += tl.dot(tl.trans(a_block), b_block)
        
        # Advance pointers
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Store the result block
    c_offsets = (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptr + c_offsets, accumulator, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_matmul_transpose(A: torch.Tensor, B: torch.Tensor):
    # A: (K, M), B: (K, N) -> C: (M, N)
    K, M = A.shape
    K_B, N = B.shape
    assert K == K_B, "K dimensions must match"

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Tuning parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_transpose_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(1), A.stride(0),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A^T * B)
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A transposed and B.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Ensure tensors are contiguous and on CUDA
        A = A.contiguous()
        B = B.contiguous()
        return triton_matmul_transpose(A, B)