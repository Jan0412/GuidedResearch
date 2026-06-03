import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Sequence length (size along dimension dim)
    stride_batch,  # Stride for batch dimension
    stride_seq,  # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch (one 1D scan)
    batch_id = tl.program_id(0)
    
    # Compute base pointers for this batch
    x_batch_ptr = x_ptr + batch_id * stride_batch
    out_batch_ptr = out_ptr + batch_id * stride_batch
    
    # Process the sequence in blocks
    for start in range(0, seq_len, BLOCK_SIZE):
        # Compute offsets for current block
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load input values
        x = tl.load(x_batch_ptr + offsets * stride_seq, mask=mask, other=0.0)
        
        # Compute cumulative sum
        cumsum = tl.cumsum(x, axis=0)
        
        # Store result
        tl.store(out_batch_ptr + offsets * stride_seq, cumsum, mask=mask)


def triton_cumsum(x: torch.Tensor, dim: int):
    """
    Triton-based cumulative sum implementation.
    
    Args:
        x: Input tensor (FP32)
        dim: Dimension along which to compute cumulative sum
        
    Returns:
        Output tensor with same shape as input
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shape and strides
    shape = x.shape
    strides = x.stride()
    
    # Normalize dimension
    dim = dim if dim >= 0 else len(shape) + dim
    
    # Calculate batch size and sequence length
    batch_size = 1
    for i in range(dim):
        batch_size *= shape[i]
    
    seq_len = shape[dim]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine block size (tuned for GPU)
    BLOCK_SIZE = 128
    
    # Create grid: one program per batch
    grid = (batch_size,)
    
    # Calculate strides for the scan dimension
    stride_batch = strides[0] if dim == 0 else 0
    for i in range(1, dim + 1):
        if i == dim:
            stride_seq = strides[i]
        else:
            stride_batch = strides[i]
    
    # Launch kernel
    cumsum_kernel[grid](
        x, out, batch_size, seq_len, stride_batch, stride_seq, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Scan model using Triton kernel for cumulative sum.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape)

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        return triton_cumsum(x, self.dim)