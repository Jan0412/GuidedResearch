import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    x_ptr, 
    out_ptr, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for LogSoftmax. 
    Processes one row per program instance.
    Implementation uses the Online Softmax algorithm to minimize memory passes.
    """
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Pointers for the start of the current row
    x_row_ptr = x_ptr + row_idx * n_cols
    out_row_ptr = out_ptr + row_idx * n_cols
    
    # --- Pass 1: Compute global max and sum of exponentials (Online Softmax) ---
    # Initialize with the first block to avoid -inf issues
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    vals = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
    
    row_max = tl.max(vals, axis=0)
    row_sum = tl.sum(tl.exp(vals - row_max), axis=0)
    
    # Iterate through the rest of the row in chunks
    for i in range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
        
        m_block = tl.max(vals, axis=0)
        s_block = tl.sum(tl.exp(vals - m_block), axis=0)
        
        # Update global max and sum using the online softmax update rule:
        # s_new = s_old * exp(m_old - m_new) + s_block * exp(m_block - m_new)
        m_new = tl.maximum(row_max, m_block)
        row_sum = row_sum * tl.exp(row_max - m_new) + s_block * tl.exp(m_block - m_new)
        row_max = m_new
        
    log_sum = tl.log(row_sum)
    
    # --- Pass 2: Compute final log_softmax values and store ---
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
        
        # LogSoftmax(x) = (x - max) - log(sum(exp(x - max)))
        out = (vals - row_max) - log_sum
        tl.store(out_row_ptr + offsets, out, mask=mask)

def triton_log_softmax(x: torch.Tensor):
    """
    Wrapper for the Triton log_softmax kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    batch_size, n_cols = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    
    # Block size for processing chunks of the row
    BLOCK_SIZE = 1024
    # Launch one program per row
    grid = (batch_size,)
    
    log_softmax_kernel[grid](
        x, out, n_cols, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs LogSoftmax activation using a custom Triton kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # The current implementation specifically optimizes for dim=1 (reduction over columns)
        # consistent with the provided architecture's batch_size, dim setup.
        return triton_log_softmax(x)