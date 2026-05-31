import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr, 
    out_ptr, 
    dim, 
    stride, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of the input tensor
    pid = tl.program_id(0)
    x_ptr += pid * stride
    out_ptr += pid * stride

    # 1. Find the maximum value in the row for numerical stability
    row_max = -float('inf')
    i = 0
    while i < dim:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load chunk and find max; use -inf for out-of-bounds to not affect max
        chunk = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(chunk, axis=0))
        i += BLOCK_SIZE

    # 2. Compute the sum of exponentials: sum(exp(x - max))
    row_sum = 0.0
    i = 0
    while i < dim:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        chunk = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        # exp(-inf) is 0, so out-of-bounds elements contribute nothing to the sum
        row_sum += tl.sum(tl.exp(chunk - row_max), axis=0)
        i += BLOCK_SIZE

    # 3. Compute the final softmax values and store them
    i = 0
    while i < dim:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        chunk = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        out_chunk = tl.exp(chunk - row_max) / row_sum
        tl.store(out_ptr + offsets, out_chunk, mask=mask)
        i += BLOCK_SIZE

def triton_softmax(x: torch.Tensor):
    """
    Triton wrapper for the softmax kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    # Stride between rows
    stride = x.stride(0)
    
    # Block size for processing the row in chunks
    BLOCK_SIZE = 1024
    
    # Grid: one program per row
    grid = (batch_size,)
    
    softmax_kernel[grid](
        x, 
        out, 
        dim, 
        stride, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softmax activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x)