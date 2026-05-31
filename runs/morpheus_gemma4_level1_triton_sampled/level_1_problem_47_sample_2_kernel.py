import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_dim1_kernel(
    x_ptr,
    out_ptr,
    M, N, K,
    s0, s1, s2,
    os0, os1, os2,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid is (M, (K + BLOCK_K - 1) // BLOCK_K)
    batch_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    # Offsets for the K dimension
    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Accumulator for the sum across the N dimension
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Loop over the reduction dimension N in chunks of BLOCK_N
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        # Compute pointers for the 2D block (BLOCK_N, BLOCK_K)
        # x[batch_id, n, k] = x_ptr + batch_id * s0 + n * s1 + k * s2
        ptr = x_ptr + batch_id * s0 + n_offsets[:, None] * s1 + k_offsets[None, :] * s2
        
        # Load the block and mask out-of-bounds elements
        vals = tl.load(ptr, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Sum across the N dimension (axis 0 of the loaded block)
        acc += tl.sum(vals, axis=0)

    # Store the result in the output tensor
    # out[batch_id, 0, k] = out_ptr + batch_id * os0 + 0 * os1 + k * os2
    out_ptr_base = out_ptr + batch_id * os0 + 0 * os1
    tl.store(out_ptr_base + k_offsets * os2, acc, mask=k_mask)


def triton_sum_dim1(x: torch.Tensor):
    """
    Optimized sum reduction over dimension 1 using Triton.
    """
    M, N, K = x.shape
    # Ensure tensor is on CUDA and contiguous for predictable strides
    x = x.contiguous()
    out = torch.empty((M, 1, K), device=x.device, dtype=x.dtype)
    
    s0, s1, s2 = x.stride()
    os0, os1, os2 = out.stride()

    BLOCK_N = 1024
    BLOCK_K = 128

    # Grid: one program per batch and one program per block of K
    grid = (M, (K + BLOCK_K - 1) // BLOCK_K)

    sum_dim1_kernel[grid](
        x, out,
        M, N, K,
        s0, s1, s2,
        os0, os1, os2,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over dimension 1 using Triton.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.
        Optimized specifically for dim=1.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        if self.dim == 1:
            return triton_sum_dim1(x)
        else:
            # Fallback to PyTorch for other dimensions to maintain functionality
            return torch.sum(x, dim=self.dim, keepdim=True)