import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batched_gemv_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    b,
    i,
    j,
    l,
    k,
    BLOCK_L: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for batched tensor-matrix multiplication.
    Computes C[b, i, j, :] = A[b, i, j, :] @ B for each batch (b, i, j).
    """
    # Grid maps to batch elements: pid corresponds to a unique (b, i, j)
    pid = tl.program_id(0)
    
    # Calculate batch indices
    batch_size = i * j
    ij = pid % batch_size
    i_idx = ij // j
    j_idx = ij % j
    
    # Pointers to current batch data
    A_ptr += pid * l
    C_ptr += pid * k
    
    # Initialize accumulator for the output vector
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    
    # Loop over reduction dimension l in chunks of BLOCK_L
    for off_l in range(0, l, BLOCK_L):
        # Load tile of A: shape (BLOCK_L,)
        # Mask to handle boundaries if l is not multiple of BLOCK_L
        offs_l = off_l + tl.arange(0, BLOCK_L)
        mask_l = offs_l < l
        A_tile = tl.load(A_ptr + offs_l, mask=mask_l, other=0.0)
        
        # Loop over output dimension k in chunks of BLOCK_K
        for off_k in range(0, k, BLOCK_K):
            # Load tile of B: shape (BLOCK_L, BLOCK_K)
            offs_k = off_k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k
            
            # B is stored as (l, k), so we need to load column-wise or use tl.load with offsets
            # B_ptr is (l, k). We want B[offs_l, offs_k]
            # We can construct the offsets for B
            B_offsets = offs_l[:, None] * k + offs_k[None, :]
            B_tile = tl.load(B_ptr + B_offsets, mask=mask_k[None, :], other=0.0)
            
            # Compute partial dot product: (BLOCK_L,) @ (BLOCK_L, BLOCK_K) -> (BLOCK_K,)
            # Using tl.dot for efficiency
            acc += tl.dot(A_tile, B_tile)
    
    # Store result
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < k
    tl.store(C_ptr + offs_k, acc, mask=mask_k)


def triton_batched_gemv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton batched GEMV kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    b, i, j, l = A.shape
    _, k = B.shape
    
    # Output tensor
    C = torch.empty((b, i, j, k), dtype=A.dtype, device=A.device)
    
    # Total number of batches
    num_batches = b * i * j
    
    # Tunable block sizes
    # l=256, k=768. Good choices: BLOCK_L=256, BLOCK_K=256 or 512
    BLOCK_L = 256
    BLOCK_K = 256
    
    # Grid configuration: 1D grid with one block per batch
    grid = (num_batches,)
    
    # Launch kernel
    batched_gemv_kernel[grid](
        A, B, C,
        b, i, j, l, k,
        BLOCK_L=BLOCK_L,
        BLOCK_K=BLOCK_K
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for batched tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication using Triton.
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
        """
        return triton_batched_gemv(A, B)


# Test code
b = 8
i = 256
j = 512
l = 256
k = 768

def get_inputs():
    A = torch.rand(b, i, j, l)
    B = torch.rand(l, k)
    return [A, B]

def get_init_inputs():
    return []