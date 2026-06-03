import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Sequence length (size of dimension 'dim')
    dim,  # Dimension along which to compute cumprod
    stride_batch,  # Stride for batch dimension
    stride_seq,  # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the batch index
    batch_idx = tl.program_id(0)
    
    # Compute base pointers for this batch
    # For simplicity, we assume dim=1 and the tensor is contiguous
    # This kernel is specialized for dim=1 to keep it simple and efficient
    x_ptr_batch = x_ptr + batch_idx * stride_batch
    out_ptr_batch = out_ptr + batch_idx * stride_batch
    
    # Load the entire sequence for this batch
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in chunks of BLOCK_SIZE
    for start in range(0, seq_len, BLOCK_SIZE):
        # Compute current offsets and mask
        curr_offsets = offsets + start
        mask = curr_offsets < seq_len
        
        # Load current values
        x_val = tl.load(x_ptr_batch + curr_offsets * stride_seq, mask=mask, other=1.0)
        
        # Compute cumulative product
        if start == 0:
            # First block: just store the values
            cumprod_val = x_val
        else:
            # Subsequent blocks: multiply by the last cumulative product from previous block
            # We need to get the cumulative product up to the start of this block
            # Load the last value from previous block
            prev_cumprod = tl.load(out_ptr_batch + (start - 1) * stride_seq)
            # Multiply all values in this block by the previous cumulative product
            cumprod_val = x_val * prev_cumprod
        
        # Store the result
        tl.store(out_ptr_batch + curr_offsets * stride_seq, cumprod_val, mask=mask)
        
        # Update cumprod_val for next iteration (element-wise cumulative product within the block)
        if start + BLOCK_SIZE < seq_len:
            # Compute running product within the block for next iteration
            # We need to update cumprod_val to be the cumulative product up to each position
            # This is done by multiplying each element with the previous cumulative product
            # We'll do this in a separate pass for simplicity
            pass  # We'll handle this in the next loop


@triton.jit
def cumprod_kernel_optimized(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Sequence length
    dim,  # Dimension along which to compute cumprod (we assume dim=1)
    stride_batch,  # Stride for batch dimension
    stride_seq,  # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the batch index
    batch_idx = tl.program_id(0)
    
    # Compute base pointers for this batch
    x_ptr_batch = x_ptr + batch_idx * stride_batch
    out_ptr_batch = out_ptr + batch_idx * stride_batch
    
    # For dim=1, we process along the sequence dimension
    # We'll use a more efficient approach with parallel prefix product
    
    # Load the entire sequence into shared memory if possible
    # But for large seq_len, we'll process in chunks
    
    # Use a single pass with running product
    running_prod = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
    
    for start in range(0, seq_len, BLOCK_SIZE):
        curr_offsets = tl.arange(0, BLOCK_SIZE) + start
        mask = curr_offsets < seq_len
        
        # Load current values
        x_val = tl.load(x_ptr_batch + curr_offsets * stride_seq, mask=mask, other=1.0)
        
        # Update running product
        if start == 0:
            running_prod = x_val
        else:
            # Load the last cumulative product from the previous block
            last_cumprod = tl.load(out_ptr_batch + (start - 1) * stride_seq)
            # Multiply all values in this block by the previous cumulative product
            running_prod = x_val * last_cumprod
        
        # Store the result
        tl.store(out_ptr_batch + curr_offsets * stride_seq, running_prod, mask=mask)


@triton.jit
def cumprod_fused_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Sequence length
    dim,  # Dimension along which to compute cumprod (we assume dim=1)
    stride_batch,  # Stride for batch dimension
    stride_seq,  # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the batch index
    batch_idx = tl.program_id(0)
    
    # Compute base pointers for this batch
    x_ptr_batch = x_ptr + batch_idx * stride_batch
    out_ptr_batch = out_ptr + batch_idx * stride_batch
    
    # For dim=1, we process along the sequence dimension
    # We'll use an efficient parallel prefix product algorithm
    
    # First, load all data into a temporary buffer if seq_len <= BLOCK_SIZE
    # Otherwise, process in chunks
    
    # Simple sequential approach for correctness (can be optimized further)
    if seq_len <= BLOCK_SIZE:
        # Single block case
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load all values
        x_vals = tl.load(x_ptr_batch + offsets * stride_seq, mask=mask, other=1.0)
        
        # Compute cumulative product sequentially
        cumprod = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
        cumprod_val = 1.0
        for i in range(seq_len):
            if i == 0:
                cumprod_val = tl.load(x_ptr_batch + i * stride_seq)
            else:
                cumprod_val = cumprod_val * tl.load(x_ptr_batch + i * stride_seq)
            cumprod = tl.store(out_ptr_batch + i * stride_seq, cumprod_val)
    else:
        # Multi-block case
        # Process first block
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load first block
        x_vals = tl.load(x_ptr_batch + offsets * stride_seq, mask=mask, other=1.0)
        
        # Compute cumulative product for first block
        cumprod_val = 1.0
        for i in range(BLOCK_SIZE):
            if i < seq_len:
                cumprod_val = cumprod_val * tl.load(x_ptr_batch + i * stride_seq)
                tl.store(out_ptr_batch + i * stride_seq, cumprod_val)
        
        # Process remaining blocks
        for start in range(BLOCK_SIZE, seq_len, BLOCK_SIZE):
            # Load current block
            curr_offsets = offsets + start
            mask = curr_offsets < seq_len
            
            x_vals = tl.load(x_ptr_batch + curr_offsets * stride_seq, mask=mask, other=1.0)
            
            # Get the last cumulative product from previous block
            last_cumprod = tl.load(out_ptr_batch + (start - 1) * stride_seq)
            
            # Compute cumulative product for current block
            block_cumprod = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
            for i in range(BLOCK_SIZE):
                if start + i < seq_len:
                    if i == 0:
                        block_cumprod = tl.load(x_ptr_batch + start * stride_seq) * last_cumprod
                        tl.store(out_ptr_batch + start * stride_seq, block_cumprod)
                    else:
                        prev_val = tl.load(out_ptr_batch + (start + i - 1) * stride_seq)
                        curr_val = tl.load(x_ptr_batch + (start + i) * stride_seq)
                        block_cumprod = prev_val * curr_val
                        tl.store(out_ptr_batch + (start + i) * stride_seq, block_cumprod)


@triton.jit
def cumprod_final_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Sequence length
    dim,  # Dimension along which to compute cumprod (we assume dim=1)
    stride_batch,  # Stride for batch dimension
    stride_seq,  # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the batch index
    batch_idx = tl.program_id(0)
    
    # Compute base pointers for this batch
    x_ptr_batch = x_ptr + batch_idx * stride_batch
    out_ptr_batch = out_ptr + batch_idx * stride_batch
    
    # For dim=1, we process along the sequence dimension
    # Use an efficient approach with a single pass
    
    # Load the first element
    if seq_len > 0:
        first_val = tl.load(x_ptr_batch)
        tl.store(out_ptr, first_val)
        
        # Process remaining elements
        for i in range(1, seq_len):
            # Load current and previous values
            curr_val = tl.load(x_ptr_batch + i * stride_seq)
            prev_cumprod = tl.load(out_ptr_batch + (i - 1) * stride_seq)
            # Compute cumulative product
            cumprod = prev_cumprod * curr_val
            # Store result
            tl.store(out_ptr_batch + i * stride_seq, cumprod)


class TritonCumprodFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        # Store input and dimension for backward pass
        ctx.save_for_backward(x)
        ctx.dim = dim
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get tensor dimensions
        shape = x.shape
        if dim < 0:
            dim = len(shape) + dim
        
        # For simplicity, assume dim=1 for now (can be extended)
        # We'll handle general dim in the kernel but specialize for dim=1
        if dim != 1:
            # Transpose to make dim=1, then transpose back
            perm = list(range(len(shape)))
            perm[1], perm[dim] = perm[dim], perm[1]
            x = x.permute(perm)
            shape = x.shape
        
        batch_size, seq_len = shape[0], shape[1]
        
        # Create output tensor
        out = torch.empty_like(x)
        
        # Calculate strides
        stride_batch = x.stride(0)
        stride_seq = x.stride(1)
        
        # Set block size
        BLOCK_SIZE = 1024
        
        # Create grid
        grid = (batch_size,)
        
        # Launch kernel
        cumprod_final_kernel[grid](
            x, out, batch_size, seq_len, dim,
            stride_batch, stride_seq, BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Transpose back if we transposed earlier
        if dim != 1:
            inv_perm = [0] * len(shape)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            out = out.permute(inv_perm)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Gradient of cumulative product: 
        # d/dx_i (prod_{j=1}^i x_j) = (prod_{j=1}^{i-1} x_j) = cumprod_{j=1}^{i-1} x_j
        # So gradient is the cumulative product of previous elements
        
        x, = ctx.saved_tensors
        dim = ctx.dim
        
        # For simplicity, implement gradient for dim=1 only
        # General case would require more complex handling
        
        # Create gradient tensor
        grad_input = torch.zeros_like(x)
        
        # Get tensor dimensions
        shape = x.shape
        if dim < 0:
            dim = len(shape) + dim
        
        # Transpose to make dim=1 if needed
        if dim != 1:
            perm = list(range(len(shape)))
            perm[1], perm[dim] = perm[dim], perm[1]
            x_t = x.permute(perm)
            grad_output_t = grad_output.permute(perm)
            grad_input_t = grad_input.permute(perm)
        else:
            x_t = x
            grad_output_t = grad_output
            grad_input_t = grad_input
        
        batch_size, seq_len = x_t.shape[0], x_t.shape[1]
        
        # Compute gradient
        for b in range(batch_size):
            cumprod_prev = 1.0
            for i in range(seq_len):
                if i == 0:
                    # Gradient for first element is 1 (since there's no previous product)
                    grad_input_t[b, i] = grad_output_t[b, i]
                else:
                    # Gradient is the product of all previous elements times gradient
                    cumprod_prev = cumprod_prev * x_t[b, i-1].item()
                    grad_input_t[b, i] = cumprod_prev * grad_output_t[b, i].item()
        
        # Transpose back if needed
        if dim != 1:
            inv_perm = [0] * len(shape)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            grad_input = grad_input_t.permute(inv_perm)
        
        return grad_input, None


class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension,
    optimized with Triton kernels.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the Optimized CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension
        using optimized Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return TritonCumprodFunction.apply(x, self.dim)