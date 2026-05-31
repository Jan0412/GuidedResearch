import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    stride_x_row,   # Stride between rows of x
    stride_out_row, # Stride between rows of out
    dim,            # Number of elements per row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_id = tl.program_id(0)
    x_row_ptr = x_ptr + row_id * stride_x_row
    out_row_ptr = out_ptr + row_id * stride_out_row

    # Pass 1: Compute the sum of absolute values for the row
    abs_sum = 0.0
    for i in range(0, tl.cdiv(dim, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load elements of the row
        vals = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        # Accumulate absolute sum
        abs_sum += tl.sum(tl.abs(vals), axis=0)

    # Calculate the mean (L1 norm divided by dimension)
    mean = abs_sum / dim
    # Use multiplicative inverse for efficiency in the second pass
    inv_mean = 1.0 / mean

    # Pass 2: Divide each element by the mean and store the result
    for i in range(0, tl.cdiv(dim, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        # Apply normalization
        out_vals = vals * inv_mean
        tl.store(out_row_ptr + offsets, out_vals, mask=mask)


def triton_l1_norm(x: torch.Tensor):
    """
    Triton wrapper for L1 normalization.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure input is contiguous for pointer arithmetic
    x = x.contiguous()
    batch_size, dim = x.shape
    out = torch.empty_like(x)

    # Block size for processing elements within a row
    BLOCK_SIZE = 1024
    # Grid is one program per row
    grid = (batch_size,)

    l1_norm_kernel[grid](
        x, 
        out, 
        x.stride(0), 
        out.stride(0), 
        dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied.
        """
        return triton_l1_norm(x)