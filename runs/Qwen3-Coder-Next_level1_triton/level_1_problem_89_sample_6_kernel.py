import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    seq_len,  # Length of sequence to cumsum over
    dim,  # Dimension along which to perform cumsum
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the batch index this program instance handles
    batch_idx = tl.program_id(0)
    
    if batch_idx >= batch_size:
        return
    
    # Compute the offset to the start of this batch
    # For dim=1, batch offset = batch_idx * seq_len
    # General formula: for a tensor of shape (..., seq_len, ...), offset = batch_idx * seq_len
    # We need to handle arbitrary dim, so we compute strides
    
    # Calculate the stride for the dimension we're cumsumming over
    # In a contiguous tensor, strides are [seq_len, 1] for dim=1 with shape (batch, seq_len)
    # For general case, we assume the tensor is contiguous and compute strides
    
    # For simplicity, we'll handle 2D case where input is (batch_size, seq_len) and dim=1
    # This matches the provided example where input is (batch_size, *input_shape) with input_shape=(32768,)
    # So effectively input is (batch_size, 32768) and dim=1
    
    # Since the example is specifically 2D (batch_size, seq_len), we optimize for that case
    if dim == 1:
        # Offset to the start of this batch in the sequence dimension
        batch_offset = batch_idx * seq_len
        
        # Initialize accumulator
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Process in chunks to handle blocks
        for start in range(0, seq_len, BLOCK_SIZE):
            end = tl.minimum(start + BLOCK_SIZE, seq_len)
            # Offsets for current position
            offsets = batch_offset + start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < (batch_offset + seq_len)
            
            # Load input values
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            
            # Update accumulator (cumulative sum)
            acc = acc + x
            
            # Store result
            tl.store(out_ptr + offsets, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_cumsum(x: torch.Tensor, dim: int = 1):
    """
    Triton implementation of cumulative sum along specified dimension.
    
    Args:
        x: Input tensor (assumed to be 2D: [batch_size, seq_len] for this implementation)
        dim: Dimension along which to compute cumsum (assumed to be 1 for this optimized version)
    
    Returns:
        Tensor with cumulative sum along specified dimension
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # For the given example, we assume 2D tensor (batch_size, seq_len)
    assert x.dim() == 2, "This implementation assumes 2D input tensor"
    assert dim == 1, "This implementation only supports dim=1"
    
    batch_size, seq_len = x.shape
    out = torch.empty_like(x)
    
    # Set block size based on sequence length
    BLOCK_SIZE = 128
    
    # Grid: one block per batch
    grid = (batch_size,)
    
    # Launch the kernel
    cumsum_kernel[grid](x, out, batch_size, seq_len, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Scan model using Triton kernel for cumulative sum.
    """

    def __init__(self, dim):
        """
        Initialize the optimized Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass for the optimized Scan model using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying cumulative sum along dim.
        """
        # For the specific use case in the example (dim=1, 2D input)
        # we can use our optimized Triton kernel
        if x.dim() == 2 and self.dim == 1:
            return triton_cumsum(x, dim=self.dim)
        else:
            # Fallback to PyTorch implementation for other cases
            return torch.cumsum(x, dim=self.dim)