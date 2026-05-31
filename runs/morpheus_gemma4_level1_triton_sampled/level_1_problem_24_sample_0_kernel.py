import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    x_ptr, 
    out_ptr, 
    stride_row, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the batch
    row_idx = tl.program_id(0)
    row_ptr = x_ptr + row_idx * stride_row
    
    # --- First Pass: Online LogSumExp ---
    # We use the online softmax trick to find the max and the sum of exponentials in one pass
    # to maintain numerical stability and reduce memory reads.
    curr_max = -1e38  # Approximate float32 minimum
    curr_sum = 0.0
    
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Load current block of the row
        vals = tl.load(row_ptr + offsets, mask=mask, other=-1e38)
        
        # Find the maximum in the current block
        local_max = tl.max(vals, axis=0)
        new_max = tl.maximum(curr_max, local_max)
        
        # Update the running sum of exponentials
        # sum_new = sum_old * exp(max_old - max_new) + sum(exp(vals - max_new))
        diff_max = curr_max - new_max
        # We use tl.where to ensure masked elements do not contribute to the sum
        sum_block = tl.sum(tl.where(mask, tl.exp(vals - new_max), 0.0), axis=0)
        curr_sum = curr_sum * tl.exp(diff_max) + sum_block
        curr_max = new_max
        
    # Compute final LogSumExp: LSE = max + log(sum)
    lse = curr_max + tl.log(curr_sum)
    
    # --- Second Pass: Compute log_softmax ---
    # log_softmax(x) = x - LSE
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(row_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + row_idx * stride_row + offsets, vals - lse, mask=mask)

def triton_log_softmax(x: torch.Tensor):
    """
    Triton wrapper for the log_softmax operation.
    Assumes reduction over the last dimension (dim=1 for 2D tensors).
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensor is contiguous for pointer arithmetic
    x = x.contiguous()
    batch_size, n_cols = x.shape
    out = torch.empty_like(x)
    
    stride_row = x.stride(0)
    # Block size for the internal reduction loop
    BLOCK_SIZE = 1024 
    
    # Grid: one program per row in the batch
    grid = (batch_size,)
    
    log_softmax_kernel[grid](
        x, out, stride_row, n_cols, 
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
        Applies LogSoftmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # The custom kernel is optimized for reduction over the last dimension (dim=1)
        return triton_log_softmax(x)