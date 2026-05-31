import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tril_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Mask for lower triangular output
    mask_out = offs_m[:, None] >= offs_n[None, :]
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Masks for triangular structure
        mask_a = offs_m[:, None] >= offs_k[None, :]
        mask_b = offs_k[:, None] >= offs_n[None, :]
        
        # Load with masking to avoid loading zeros
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, 
                    mask=mask_a & mask_m[:, None], other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, 
                    mask=mask_b & mask_n[None, :], other=0.0)
        
        # Compute dot product
        acc += tl.dot(a, b)
    
    # Store only lower triangular elements
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, 
             acc, mask=mask_out & mask_m[:, None] & mask_n[None, :])


def triton_tril_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match."
    
    C = torch.empty((M, N), dtype=torch.float32, device='cuda')
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_stages = 2
    
    # Grid calculation
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Strides
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    
    # Launch kernel
    tril_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_stages=num_stages
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, A, B):
        return triton_tril_matmul(A, B)


def get_inputs():
    M = 4096
    A = torch.rand(M, M, device='cuda')
    B = torch.rand(M, M, device='cuda')
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]


def get_init_inputs():
    return []