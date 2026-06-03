import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr,      # Pointer to input tensor
    mask_ptr,   # Pointer to mask tensor
    out_ptr,    # Pointer to output tensor
    batch_size, # Number of batches
    seq_len,    # Sequence length (size along dimension dim)
    stride_batch, # Stride for batch dimension
    stride_seq,   # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch
    batch_id = tl.program_id(0)
    
    # Calculate base pointers for this batch
    x_offset = batch_id * stride_batch
    mask_offset = batch_id * stride_batch
    out_offset = batch_id * stride_batch
    
    # Initialize running sum
    cumsum = tl.zeros([1], dtype=tl.float32)
    
    # Process sequence in blocks
    for start in range(0, seq_len, BLOCK_SIZE):
        # Compute offsets for current block
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load values
        x_vals = tl.load(x_ptr + x_offset + offsets * stride_seq, mask=mask, other=0.0)
        mask_vals = tl.load(mask_ptr + mask_offset + offsets * stride_seq, mask=mask, other=0)
        
        # Convert mask to float for computation
        mask_float = mask_vals.to(tl.float32)
        
        # Compute masked values
        masked_vals = x_vals * mask_float
        
        # Update cumulative sum
        cumsum = cumsum + masked_vals
        
        # Store result
        tl.store(out_ptr + out_offset + offsets * stride_seq, cumsum, mask=mask)


class TritonMaskedCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, dim):
        # Ensure inputs are contiguous
        x = x.contiguous()
        mask = mask.contiguous()
        
        # Get tensor shapes
        batch_size = 1
        seq_len = x.shape[dim]
        
        # Calculate batch size (product of all dimensions except dim)
        for i, s in enumerate(x.shape):
            if i != dim:
                batch_size *= s
        
        # Calculate strides
        stride_batch = x.stride(dim) if dim == 0 else x.stride(0) // x.stride(dim) * x.shape[dim]
        # Normalize strides to get actual strides
        strides = list(x.stride())
        stride_batch = strides[dim] if dim == 0 else strides[0] // strides[dim] * x.shape[dim]
        
        # Recalculate proper strides
        stride_batch = 1
        for i in range(dim):
            stride_batch *= x.shape[i]
        stride_seq = x.stride(dim)
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Determine block size (tuned for sequential processing)
        BLOCK_SIZE = min(128, seq_len)
        
        # Grid: one block per batch
        grid = (batch_size,)
        
        # Launch kernel
        masked_cumsum_kernel[grid](
            x, mask, out,
            batch_size, seq_len,
            stride_batch, stride_seq,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        ctx.save_for_backward(x, mask)
        ctx.dim = dim
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        dim = ctx.dim
        
        # For backward pass, we need to compute gradient of cumsum
        # grad_input = grad_output (since d/dx(cumsum(x)) = 1 for elements in cumsum)
        # But we need to mask it appropriately
        
        grad_input = grad_output * mask
        
        return grad_input, None, None


def triton_masked_cumsum(x, mask, dim):
    return TritonMaskedCumsum.apply(x, mask, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using custom Triton kernel.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        # Move dim to position 1 for easier processing if not already
        original_shape = x.shape
        if self.dim != 1:
            # Permute to make dim=1, process, then permute back
            dims = list(range(len(original_shape)))
            dims.remove(self.dim)
            dims.insert(1, self.dim)
            x = x.permute(dims)
            mask = mask.permute(dims)
        
        # Call our optimized function
        result = triton_masked_cumsum(x, mask, dim=1)
        
        # Permute back if we moved dimensions
        if self.dim != 1:
            dims_back = list(range(len(original_shape)))
            dims_back.remove(1)
            dims_back.insert(self.dim, 1)
            result = result.permute(dims_back)
        
        return result