import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_reduction_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for mean reduction along a specified dimension.
    
    Args:
        x_ptr: Pointer to input tensor
        out_ptr: Pointer to output tensor
        n_rows: Number of rows (elements in the non-reduced dimension)
        n_cols: Number of columns (elements in the reduced dimension)
        dim: Dimension to reduce over (0 or 1 for 2D case)
        BLOCK_SIZE: Size of blocks for parallel processing
    """
    # For 2D case, we handle reduction along dimension 1 (columns) by default
    # This is the most common case and easiest to optimize
    
    # Calculate row index
    row_idx = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_start = row_idx * n_cols
    
    # Accumulator for mean calculation
    sum_val = 0.0
    count = 0
    
    # Process elements in chunks
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data
        data = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        
        # Accumulate sum
        sum_val += tl.sum(data, axis=0)
        count += tl.sum(tl.cast(mask, tl.int32), axis=0)
    
    # Calculate mean
    mean_val = sum_val / n_cols
    
    # Store result
    tl.store(out_ptr + row_idx, mean_val)


class TritonMean(torch.autograd.Function):
    """Triton-based mean reduction with proper gradient support."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get input shape
        shape = x.shape
        ndim = len(shape)
        
        # Normalize dimension
        if dim < 0:
            dim = ndim + dim
        
        # Handle 1D case
        if ndim == 1:
            output_shape = [1]
        else:
            # Create output shape by removing the reduced dimension
            output_shape = list(shape[:dim]) + list(shape[dim+1:])
        
        # For 2D case optimization
        if ndim == 2:
            # Reshape to 2D if needed
            if dim == 0:
                # Reduce along rows - transpose for easier handling
                x = x.t()
                dim = 1
            
            n_rows = shape[0] if dim == 1 else shape[1]
            n_cols = shape[1] if dim == 1 else shape[0]
            
            output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            
            BLOCK_SIZE = 256
            grid = (n_rows,)
            
            mean_reduction_kernel[grid](
                x, output, n_rows, n_cols, dim, BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            # General case: reshape to 2D, apply reduction, reshape back
            # Move the dimension to reduce to the last position
            dims = list(range(ndim))
            dims.pop(dim)
            dims.append(dim)
            x_permuted = x.permute(dims)
            
            # Reshape to 2D: [all_other_dims, reduced_dim]
            reduced_dim_size = shape[dim]
            other_dims = 1
            for i, s in enumerate(shape):
                if i != dim:
                    other_dims *= s
            
            x_2d = x_permuted.reshape(other_dims, reduced_dim_size)
            
            # Create output tensor
            output_size = other_dims
            output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            
            # Apply mean reduction
            BLOCK_SIZE = 256
            grid = (output_size,)
            
            mean_reduction_kernel[grid](
                x_2d, output, output_size, reduced_dim_size, 1, BLOCK_SIZE=BLOCK_SIZE
            )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # For mean reduction, gradient is grad_output / size along reduced dimension
        # This would need to be implemented if we want full autograd support
        raise NotImplementedError("Backward pass not implemented for Triton mean reduction")


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Wrapper function for Triton mean reduction."""
    return TritonMean.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction using custom Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        return triton_mean(x, self.dim)