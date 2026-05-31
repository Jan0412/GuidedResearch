import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduce_kernel(
    x_ptr, 
    out_ptr, 
    B, D1, D2, 
    stride_b, stride_d1, stride_d2, 
    stride_b_out, stride_d2_out, 
    BLOCK_SIZE_D1: tl.constexpr, 
    BLOCK_SIZE_D2: tl.constexpr,
):
    # Each program handles one block of the reduction (one batch element and a chunk of the D2 dimension)
    pid_b = tl.program_id(0)
    pid_d2 = tl.program_id(1)

    # Offsets for the D2 dimension
    offs_d2 = pid_d2 * BLOCK_SIZE_D2 + tl.arange(0, BLOCK_SIZE_D2)
    mask_d2 = offs_d2 < D2

    # Accumulator for the sum over D1
    acc = tl.zeros([BLOCK_SIZE_D2], dtype=tl.float32)

    # Iterate over the reduction dimension D1 in blocks
    for d1_start in range(0, D1, BLOCK_SIZE_D1):
        offs_d1 = d1_start + tl.arange(0, BLOCK_SIZE_D1)
        mask_d1 = offs_d1 < D1

        # Calculate pointers for the block (BLOCK_SIZE_D1, BLOCK_SIZE_D2)
        # x_ptr + batch_offset + d1_offset + d2_offset
        ptr = x_ptr + pid_b * stride_b + offs_d1[:, None] * stride_d1 + offs_d2[None, :] * stride_d2
        
        # Load the block with masking
        val = tl.load(ptr, mask=mask_d1[:, None] & mask_d2[None, :], other=0.0)
        
        # Reduce along the D1 axis (axis 0 of the loaded block)
        acc += tl.sum(val, axis=0)

    # Store the result in the output tensor
    out_ptr_final = out_ptr + pid_b * stride_b_out + offs_d2 * stride_d2_out
    tl.store(out_ptr_final, acc, mask=mask_d2)


def triton_sum(x: torch.Tensor, dim: int):
    """
    Triton wrapper for sum reduction along dim=1 for 3D tensors.
    """
    # This specific optimization is targeted at the provided architecture (dim=1)
    if dim != 1 or x.ndim != 3:
        return torch.sum(x, dim=dim, keepdim=True)

    # Ensure input is contiguous for predictable striding
    x = x.contiguous()
    B, D1, D2 = x.shape
    
    # Output shape is (B, 1, D2)
    out = torch.empty((B, 1, D2), dtype=x.dtype, device=x.device)

    stride_b, stride_d1, stride_d2 = x.stride()
    stride_b_out, _, stride_d2_out = out.stride()

    # Tunable block sizes
    BLOCK_SIZE_D1 = 128
    BLOCK_SIZE_D2 = 32

    # Grid: (Batch, D2_chunks)
    grid = (B, (D2 + BLOCK_SIZE_D2 - 1) // BLOCK_SIZE_D2)

    sum_reduce_kernel[grid](
        x, out, B, D1, D2, 
        stride_b, stride_d1, stride_d2, 
        stride_b_out, stride_d2_out, 
        BLOCK_SIZE_D1=BLOCK_SIZE_D1, 
        BLOCK_SIZE_D2=BLOCK_SIZE_D2
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
        return triton_sum(x, self.dim)