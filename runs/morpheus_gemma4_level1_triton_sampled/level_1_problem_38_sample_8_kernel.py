import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr, 
    out_ptr, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to perform L1 normalization along the second dimension.
    Each program handles one row of the input tensor.
    """
    # Identify the row this program is responsible for
    row_idx = tl.program_id(0)
    # Offset to the start of the row
    row_offset = row_idx * n_cols

    # Pass 1: Compute the sum of absolute values for the row
    sum_abs = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Load elements, filling out-of-bounds with 0.0
        vals = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        # Accumulate the absolute values
        sum_abs += tl.sum(tl.abs(vals), axis=0)

    # Calculate the mean (L1 norm divided by the number of elements)
    mean = sum_abs / n_cols

    # Pass 2: Divide each element by the mean and store the result
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        # Perform the normalization
        norm_vals = vals / mean
        # Store the result back to global memory
        tl.store(out_ptr + row_offset + offsets, norm_vals, mask=mask)

def triton_l1_norm(x: torch.Tensor):
    """
    Wrapper for the l1_norm_kernel.
    """
    # Ensure input is on GPU and contiguous
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Tunable BLOCK_SIZE (must be a power of 2)
    BLOCK_SIZE = 1024
    
    # Grid: one program per row
    grid = (n_rows,)
    
    # Launch the kernel
    l1_norm_kernel[grid](
        x, 
        out, 
        n_cols, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using a fused Triton kernel.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using the custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied.
        """
        return triton_l1_norm(x)