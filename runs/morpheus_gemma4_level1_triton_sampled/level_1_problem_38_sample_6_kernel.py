import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr,
    out_ptr,
    stride_x_row,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for L1 normalization: x / mean(abs(x), dim=1)
    Each program handles one row of the input tensor.
    """
    # Get the row index this program is responsible for
    row_idx = tl.program_id(0)
    # Offset to the start of the current row
    row_offset = row_idx * stride_x_row

    # First pass: Compute the sum of absolute values for the row
    abs_sum = 0.0
    for i in range(0, tl.cdiv(dim, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load values and compute absolute sum
        vals = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        abs_sum += tl.sum(tl.abs(vals), axis=0)

    # Compute the multiplication factor: 1 / (abs_sum / dim) = dim / abs_sum
    # We use this to replace division with multiplication in the second pass
    factor = dim / abs_sum

    # Second pass: Apply the normalization factor to each element in the row
    for i in range(0, tl.cdiv(dim, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + row_offset + offsets, vals * factor, mask=mask)


def triton_l1_norm(x: torch.Tensor):
    """
    Wrapper for the Triton L1 normalization kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure the tensor is contiguous for pointer arithmetic
    x = x.contiguous()
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    # Calculate stride for moving between rows
    stride_x_row = x.stride(0)

    # Block size for processing chunks of the row
    BLOCK_SIZE = 1024
    # Grid is defined by the batch size (one program per row)
    grid = (batch_size,)

    l1_norm_kernel[grid](
        x, 
        out, 
        stride_x_row, 
        dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using a custom Triton kernel.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Replace the PyTorch implementation with the optimized Triton kernel
        return triton_l1_norm(x)