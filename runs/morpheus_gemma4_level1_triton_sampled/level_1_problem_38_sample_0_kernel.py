import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    stride_x,       # Stride between rows of input
    stride_out,     # Stride between rows of output
    dim,            # Dimension size (number of elements per row)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input matrix
    pid = tl.program_id(0)
    
    # Pointers to the start of the current row
    row_x_ptr = x_ptr + pid * stride_x
    row_out_ptr = out_ptr + pid * stride_out

    # Pass 1: Compute the sum of absolute values for the row
    acc = 0.0
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load values, masking out-of-bounds elements with 0.0
        val = tl.load(row_x_ptr + offsets, mask=mask, other=0.0)
        # Accumulate absolute values
        acc += tl.sum(tl.abs(val), axis=0)

    # Calculate the mean of absolute values
    mean = acc / dim

    # Pass 2: Divide the original values by the mean and store the result
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        val = tl.load(row_x_ptr + offsets, mask=mask, other=0.0)
        # Perform L1 normalization
        tl.store(row_out_ptr + offsets, val / mean, mask=mask)


def triton_l1_norm(x: torch.Tensor):
    """
    Wrapper for the Triton L1 normalization kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure input is contiguous for predictable striding
    x = x.contiguous()
    out = torch.empty_like(x)
    
    batch_size, dim = x.shape
    stride_x = x.stride(0)
    stride_out = out.stride(0)
    
    # Use a block size that is a power of 2
    BLOCK_SIZE = 1024
    
    # Grid is defined by the batch size (one program per row)
    grid = (batch_size,)
    
    l1_norm_kernel[grid](
        x, 
        out, 
        stride_x, 
        stride_out, 
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
        Applies L1 normalization to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return triton_l1_norm(x)