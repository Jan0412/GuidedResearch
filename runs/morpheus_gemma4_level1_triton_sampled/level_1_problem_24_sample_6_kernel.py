import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    x_ptr, 
    out_ptr, 
    stride_x_row, 
    stride_out_row, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the row for input and output
    x_row_ptr = x_ptr + row_idx * stride_x_row
    out_row_ptr = out_ptr + row_idx * stride_out_row
    
    # --- First Pass: Compute LogSumExp using Online Softmax technique ---
    # Initialize with the first block to avoid NaN with -inf
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x_first = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
    
    m_i = tl.max(x_first)
    s_i = tl.sum(tl.exp(x_first - m_i))
    
    # Loop through the rest of the row in blocks
    for start in range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
        
        m_local = tl.max(x)
        m_next = tl.maximum(m_i, m_local)
        
        # Online sum update: s_next = s_i * exp(m_i - m_next) + sum(exp(x - m_next))
        s_i = s_i * tl.exp(m_i - m_next) + tl.sum(tl.exp(x - m_next))
        m_i = m_next
        
    # Final LogSumExp value for the row
    lse = m_i + tl.log(s_i)
    
    # --- Second Pass: Compute x - LSE and store result ---
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
        tl.store(out_row_ptr + offsets, x - lse, mask=mask)

def triton_log_softmax(x: torch.Tensor, dim: int):
    """
    Triton wrapper for LogSoftmax. 
    Optimized for 2D tensors where reduction is over the last dimension (dim=1).
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure the tensor is contiguous for pointer arithmetic
    x = x.contiguous()
    
    batch_size, n_cols = x.shape
    out = torch.empty_like(x)
    
    # Strides for row-major access
    stride_x_row = x.stride(0)
    stride_out_row = out.stride(0)
    
    # Block size for the reduction loop. 1024 is typically efficient for FP32.
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    log_softmax_kernel[grid](
        x, 
        out, 
        stride_x_row, 
        stride_out_row, 
        n_cols, 
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
        Applies LogSoftmax activation to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # We assume the reduction is over the last dimension (dim=1) as per original architecture
        return triton_log_softmax(x, self.dim)