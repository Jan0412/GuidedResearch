import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    X,  # Input tensor pointer
    Out,  # Output tensor pointer (indices)
    M,  # Number of rows (or outer dimension size)
    N,  # Number of columns (or dimension size we're reducing)
    stride_m,  # Stride between rows (or outer dimension elements)
    stride_n,  # Stride within the dimension we're reducing
    BLOCK_SIZE: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # Get the row index this program handles
    row_idx = tl.program_id(0)
    
    # Calculate base pointer for this row
    X_ptr = X + row_idx * stride_m
    Out_ptr = Out + row_idx
    
    # Initialize max value and index
    max_val = tl.zeros((1,), dtype=tl.float32)
    max_idx = tl.zeros((1,), dtype=tl.int64)
    
    # Process in chunks of BLOCK_SIZE
    for start_n in range(0, N, BLOCK_SIZE):
        # Calculate offsets for this chunk
        n_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid indices
        if EVEN_N:
            mask = tl.full((BLOCK_SIZE,), True, dtype=tl.bool)
        else:
            mask = n_offsets < N
        
        # Load data
        x = tl.load(X_ptr + n_offsets * stride_n, mask=mask, other=-float('inf'))
        
        # Convert to float32 for computation
        x_f32 = x.to(tl.float32)
        
        # Find local max
        local_max = tl.max(x_f32)
        local_max_idx = tl.argmax(x_f32, axis=0)  # Returns index within the block
        
        # Update global max and index
        # If local max is greater than current max, update
        condition = local_max > max_val
        max_val = tl.where(condition, local_max, max_val)
        max_idx = tl.where(condition, start_n + local_max_idx, max_idx)
    
    # Store the result
    tl.store(Out_ptr, max_idx.to(tl.int64))


class TritonArgmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        original_shape = x.shape
        dim = dim % len(original_shape)
        
        # Reshape to 2D for easier processing: [outer_dims, reduce_dim]
        if dim == len(original_shape) - 1:
            # Last dimension case: reshape to [N, M] where N is product of all dims before last, M is last dim
            outer_size = 1
            for i in range(len(original_shape) - 1):
                outer_size *= original_shape[i]
            M, N = outer_size, original_shape[-1]
            stride_m = N
            stride_n = 1
        else:
            # General case: move the target dimension to the end, then reshape
            permute_dims = list(range(len(original_shape)))
            permute_dims.pop(dim)
            permute_dims.append(dim)
            x_permuted = x.permute(permute_dims)
            x_permuted = x_permuted.contiguous()
            
            # Now reshape to 2D: [outer_dims, reduce_dim]
            outer_size = 1
            for i in range(len(original_shape) - 1):
                outer_size *= x_permuted.shape[i]
            M, N = outer_size, x_permuted.shape[-1]
            stride_m = N
            stride_n = 1
            
            # Store the permuted shape for potential backward (not needed for argmax)
            ctx.save_for_backward(torch.tensor(permute_dims))
            ctx.original_shape = original_shape
            ctx.dim = dim
            x = x_permuted
        
        # Create output tensor
        # The output shape should have the specified dimension removed
        output_shape = list(original_shape)
        output_shape.pop(dim)
        out = torch.empty(output_shape, dtype=torch.long, device=x.device)
        
        # Set block size (tunable)
        BLOCK_SIZE = 256
        
        # Determine if N is divisible by BLOCK_SIZE for optimization
        EVEN_N = (N % BLOCK_SIZE == 0)
        
        # Launch kernel
        grid = (M, 1, 1)
        argmax_kernel[grid](x, out, M, N, stride_m, stride_n, BLOCK_SIZE=BLOCK_SIZE, EVEN_N=EVEN_N)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Argmax is not differentiable in the usual sense, so we return None gradients
        return None, None


def triton_argmax(x: torch.Tensor, dim: int):
    """
    Wrapper function for Triton-based argmax operation.
    """
    return TritonArgmax.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return triton_argmax(x, self.dim)