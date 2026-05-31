import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def masked_cumsum_kernel(
    x_ptr, mask_ptr, out_ptr,
    stride_x_red, stride_x_batch,
    stride_m_red, stride_m_batch,
    stride_o_red, stride_o_batch,
    n_red,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one "line" along the reduction dimension
    batch_idx = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask_range = offsets < n_red
    
    # Calculate base pointers for the current line
    # We shift the pointer by the batch index * batch stride
    x_row_ptr = x_ptr + batch_idx * stride_x_batch
    m_row_ptr = mask_ptr + batch_idx * stride_m_batch
    o_row_ptr = out_ptr + batch_idx * stride_o_batch
    
    # Load data along the reduction dimension
    x = tl.load(x_row_ptr + offsets * stride_x_red, mask=mask_range)
    m = tl.load(m_row_ptr + offsets * stride_m_red, mask=mask_range)
    
    # Apply the mask: elements where mask is False become 0.0
    # Triton handles float * bool by casting bool to 0.0/1.0
    val = x * m
    
    # Perform cumulative sum along the reduction axis
    res = tl.cumsum(val, axis=0)
    
    # Store the result back to memory
    tl.store(o_row_ptr + offsets * stride_o_red, res, mask=mask_range)

def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Triton wrapper for masked cumulative sum.
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    
    # Ensure inputs are contiguous to avoid complex stride calculations 
    # and ensure the reduction dimension is handled correctly.
    x = x.contiguous()
    mask = mask.contiguous()
    
    out = torch.empty_like(x)
    shape = x.shape
    n_red = shape[dim]
    
    # Calculate total number of parallel lines (product of all non-reduction dimensions)
    batch_shape = [shape[i] for i in range(len(shape)) if i != dim]
    batch_size = 1
    for s in batch_shape:
        batch_size *= s
        
    # Strides for the reduction dimension
    stride_x_red = x.stride(dim)
    stride_m_red = mask.stride(dim)
    stride_o_red = out.stride(dim)
    
    # For 2D tensors, we can easily identify the batch stride.
    # For N-D tensors, we flatten the batch dimensions.
    # To make this generic for N-D without expensive transposes, 
    # we can treat the tensor as a flattened array and calculate the 
    # offset of the "line" manually, but for the given architecture (2D), 
    # we can use the stride of the other dimension.
    if len(shape) == 2:
        stride_x_batch = x.stride(0) if dim == 1 else x.stride(1)
        stride_m_batch = mask.stride(0) if dim == 1 else mask.stride(1)
        stride_o_batch = out.stride(0) if dim == 1 else out.stride(1)
    else:
        # Fallback for N-D: transpose the reduction dim to the end, 
        # flatten the rest, and then we can use a simple batch stride.
        # However, given the specific problem constraints, we optimize for the 2D case.
        # To support N-D, we'd typically reshape/transpose.
        x_flat = x.transpose(dim, -1).reshape(-1, n_red).contiguous()
        m_flat = mask.transpose(dim, -1).reshape(-1, n_red).contiguous()
        o_flat = torch.empty_like(x_flat)
        
        stride_x_red, stride_x_batch = 1, n_red
        stride_m_red, stride_m_batch = 1, n_red
        stride_o_red, stride_o_batch = 1, n_red
        
        # Re-bind pointers to the flattened versions
        x_ptr, mask_ptr, out_ptr = x_flat, m_flat, o_flat
    else:
        x_ptr, mask_ptr, out_ptr = x, mask, out

    # Define BLOCK_SIZE as the next power of 2 of the reduction dimension
    BLOCK_SIZE = 1 << (n_red - 1).bit_length()
    
    grid = (batch_size,)
    
    # Launch kernel
    # If we used the N-D fallback, we use the flattened tensors
    if len(shape) > 2:
        masked_cumsum_kernel[grid](
            x_flat, m_flat, o_flat,
            stride_x_red, stride_x_batch,
            stride_m_red, stride_m_batch,
            stride_o_red, stride_o_batch,
            n_red,
            BLOCK_SIZE=BLOCK_SIZE
        )
        # Reshape and transpose back to original shape
        return o_flat.reshape(batch_shape + [n_red]).transpose(len(batch_shape)-1, dim)
    else:
        masked_cumsum_kernel[grid](
            x, mask, out,
            stride_x_red, stride_x_batch,
            stride_m_red, stride_m_batch,
            stride_o_red, stride_o_batch,
            n_red,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using a custom Triton kernel.
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
        return triton_masked_cumsum(x, mask, self.dim)