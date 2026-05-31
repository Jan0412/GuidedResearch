import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    x_ptr, 
    out_ptr, 
    batch_size, 
    dim, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row of the input tensor
    row_idx = tl.program_id(0)
    if row_idx >= batch_size:
        return

    # Pointers to the start of the current row for input and output
    x_row_ptr = x_ptr + row_idx * dim
    out_row_ptr = out_ptr + row_idx * dim

    # --- Pass 1: Compute Max and Sum of Exponents (Online Softmax) ---
    # We use the online softmax algorithm to avoid numerical instability and 
    # handle the large dimension size by iterating in blocks.
    max_val = -float('inf')
    sum_val = 0.0
    
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load block of values. Masked elements are set to -inf so they don't affect the max.
        vals = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
        
        # Compute the maximum of the current block
        local_max = tl.max(vals)
        
        # Update the global maximum for the row
        new_max = tl.maximum(max_val, local_max)
        
        # Rescale the previous sum to the new maximum and add the current block's sum
        # sum_new = sum_old * exp(max_old - max_new) + sum(exp(x_i - max_new))
        sum_val = sum_val * tl.exp(max_val - new_max) + tl.sum(tl.exp(vals - new_max))
        max_val = new_max

    # Log-Sum-Exp (LSE) = max + log(sum(exp(x - max)))
    log_sum_exp = max_val + tl.log(sum_val)

    # --- Pass 2: Compute LogSoftmax elements ---
    # log_softmax(x) = x - LSE
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        vals = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_row_ptr + offsets, vals - log_sum_exp, mask=mask)

def triton_log_softmax(x: torch.Tensor):
    """
    Wrapper function to launch the Triton log_softmax kernel.
    """
    # Ensure input is on CUDA and contiguous
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    # BLOCK_SIZE is a tunable parameter. 1024 is generally efficient for FP32.
    BLOCK_SIZE = 1024
    
    # Grid is one program per row
    grid = (batch_size,)
    
    log_softmax_kernel[grid](
        x, 
        out, 
        batch_size, 
        dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a LogSoftmax activation using a custom Triton kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using the Triton implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # The custom kernel is optimized for reduction along the last dimension (dim=1 for 2D)
        return triton_log_softmax(x)