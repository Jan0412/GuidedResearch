import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduction_kernel(
    x_ptr,
    out_ptr,
    stride_b,
    stride_d1,
    stride_d2,
    out_stride_b,
    out_stride_d2,
    B,
    D1,
    D2,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_d2_block = tl.program_id(1)

    # Offset for the feature dimension (D2)
    offsets_d2 = pid_d2_block * BLOCK_D2 + tl.arange(0, BLOCK_D2)
    mask_d2 = offsets_d2 < D2

    # Accumulator for the sum reduction over D1
    acc = tl.zeros((BLOCK_D2,), dtype=tl.float32)

    # Loop over the reduction dimension (D1) in blocks
    for i in range(0, D1, BLOCK_D1):
        offsets_d1 = i + tl.arange(0, BLOCK_D1)
        mask_d1 = offsets_d1 < D1
        
        # Compute pointers for the current block: shape (BLOCK_D1, BLOCK_D2)
        # x[pid_b, offsets_d1, offsets_d2]
        ptr = x_ptr + pid_b * stride_b + offsets_d1[:, None] * stride_d1 + offsets_d2[None, :] * stride_d2
        
        # Load data with boundary masking
        vals = tl.load(ptr, mask=mask_d1[:, None] & mask_d2[None, :], other=0.0)
        
        # Sum along the BLOCK_D1 dimension
        acc += tl.sum(vals, axis=0)

    # Compute output pointer for the current block of D2
    # out[pid_b, 0, offsets_d2]
    out_ptr_final = out_ptr + pid_b * out_stride_b + offsets_d2 * out_stride_d2
    tl.store(out_ptr_final, acc, mask=mask_d2)


def triton_sum_dim1(x: torch.Tensor):
    """
    Triton wrapper for torch.sum(x, dim=1, keepdim=True)
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensor is contiguous for predictable striding
    x = x.contiguous()
    B, D1, D2 = x.shape
    
    # Output shape (B, 1, D2)
    out = torch.empty((B, 1, D2), dtype=x.dtype, device=x.device)
    
    # Strides
    stride_b, stride_d1, stride_d2 = x.stride()
    out_stride_b, _, out_stride_d2 = out.stride()

    # Tuning parameters
    BLOCK_D1 = 1024
    BLOCK_D2 = 32

    # Grid: one program per batch and per block of D2
    grid = (B, (D2 + BLOCK_D2 - 1) // BLOCK_D2)

    sum_reduction_kernel[grid](
        x, out,
        stride_b, stride_d1, stride_d2,
        out_stride_b, out_stride_d2,
        B, D1, D2,
        BLOCK_D1=BLOCK_D1,
        BLOCK_D2=BLOCK_D2,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension using Triton.
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

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # Optimization specifically for the provided case where dim=1
        if self.dim == 1 and x.ndim == 3:
            return triton_sum_dim1(x)
        else:
            # Fallback for other dimensions or shapes
            return torch.sum(x, dim=self.dim, keepdim=True)