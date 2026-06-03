import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    batch_size,
    seq_len,
    dim,
    BLOCK_SIZE: tl.constexpr,
    MASK_VALUE: tl.constexpr,
):
    # Determine which batch we're processing
    batch_id = tl.program_id(0)
    
    # Calculate offsets for this batch
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Base pointer for this batch
    base_offset = batch_id * seq_len
    
    # Initialize running sum
    running_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process in chunks if needed
    num_chunks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * BLOCK_SIZE
        chunk_offsets = chunk_start + offsets
        mask = chunk_offsets < seq_len
        
        # Load x values
        x = tl.load(x_ptr + base_offset + chunk_offsets, mask=mask, other=0.0)
        
        # Load mask values
        mask_val = tl.load(mask_ptr + base_offset + chunk_offsets, mask=mask, other=0)
        mask_val = mask_val.to(tl.float32)
        
        # Apply mask and accumulate
        masked_x = x * mask_val
        running_sum = running_sum + masked_x
        
        # Store result
        tl.store(out_ptr + base_offset + chunk_offsets, running_sum.to(x_ptr.dtype.element_ty), mask=mask)


class TritonMaskedCumsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, dim):
        # Ensure inputs are contiguous
        x = x.contiguous()
        mask = mask.contiguous()
        
        # Store for backward pass
        ctx.save_for_backward(x, mask)
        ctx.dim = dim
        
        # Create output tensor
        out = torch.empty_like(x)
        
        # Get dimensions
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # For 1D case along dimension 1
        if dim == 1:
            BLOCK_SIZE = 256
            grid = (batch_size,)
            
            # Launch kernel
            masked_cumsum_kernel[grid](
                x, mask, out,
                batch_size, seq_len, dim,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            # Fallback to PyTorch for other dimensions
            out = torch.cumsum(x * mask, dim=dim)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Gradient of masked cumulative sum is just the gradient passed through
        # where mask is True, otherwise 0
        x, mask = ctx.saved_tensors
        dim = ctx.dim
        
        # Gradient w.r.t. x is grad_output where mask is True, else 0
        grad_x = grad_output * mask if mask.dtype == torch.bool else grad_output * mask.to(torch.bool)
        
        # Gradient w.r.t. mask is x * grad_output where mask is True
        grad_mask = x * grad_output if mask.dtype == torch.bool else x * grad_output * mask.to(torch.bool)
        
        return grad_x, grad_mask, None


def triton_masked_cumsum(x, mask, dim):
    return TritonMaskedCumsumFunction.apply(x, mask, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using Triton kernels.

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
        # Handle the case where dim != 1 by falling back to PyTorch
        if self.dim != 1:
            return torch.cumsum(x * mask, dim=self.dim)
        
        # Use optimized Triton kernel for dim=1
        return triton_masked_cumsum(x, mask, self.dim)