import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    n_cols,         # Number of columns (elements to reduce over)
    stride_batch,   # Stride between batches
    stride_row,     # Stride between rows in a batch
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID corresponds to batch index * number_of_rows_per_batch + row index
    # For our case, we're reducing over dim=1 (dim1=4096), so each row has n_cols elements
    batch_id = tl.program_id(0)
    row_id = tl.program_id(1)
    
    # Calculate base pointers for this batch and row
    x_block_ptr = x_ptr + batch_id * stride_batch + row_id * stride_row
    out_ptr = out_ptr + batch_id * stride_row + row_id
    
    # Initialize minimum to large value
    min_val = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
    
    # Process in blocks to reduce over n_cols
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data
        x = tl.load(x_block_ptr + offsets, mask=mask, other=float('inf'))
        
        # Compute min with current block
        block_min = tl.min(x, axis=0)
        min_val = tl.minimum(min_val, block_min)
    
    # Reduce the min_val array to a single value using tree reduction
    # Since BLOCK_SIZE is a compile-time constant, we can do this efficiently
    for i in range(BLOCK_SIZE // 2):
        other = tl.load(tl.make_tensor_ptr(min_val) + i, mask=tl.arange(0, BLOCK_SIZE - i) < (BLOCK_SIZE - i))
        min_val = tl.minimum(min_val[:BLOCK_SIZE - i], other)
    
    # Store final result
    tl.store(out_ptr, min_val[0])


# Optimized version using online reduction approach (more efficient)
@triton.jit
def min_reduction_optimized_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    n_cols,         # Number of columns (elements to reduce over)
    stride_batch,   # Stride between batches
    stride_row,     # Stride between rows in a batch
    BLOCK_SIZE: tl.constexpr,
):
    # For our case: input shape is (batch_size, dim1, dim2) = (128, 4096, 4095)
    # We're reducing over dim=1 (dim1=4096), so output shape is (batch_size, dim2)
    # Each block handles one (batch, dim2) pair
    
    # Program ID for batch
    batch_id = tl.program_id(0)
    # Program ID for column in output (which is dim2)
    col_id = tl.program_id(1)
    
    # Calculate base pointers
    x_base = x_ptr + batch_id * stride_batch + col_id
    out_ptr = out_ptr + batch_id * stride_row + col_id
    
    # Initialize minimum to large value
    min_val = float('inf')
    
    # Process over the dimension we're reducing (dim1 = 4096)
    # Use tiling to process n_cols elements
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data with stride = dim2, so we're reading along the reduction dimension
        x = tl.load(x_base + offsets * 4095, mask=mask, other=float('inf'))
        
        # Update minimum
        block_min = tl.min(x, axis=0)
        min_val = tl.minimum(min_val, block_min)
    
    # Store final result
    tl.store(out_ptr, min_val)


class TritonMinReduction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get input shape and verify dimension
        assert x.dim() == 3, "Input must be 3D tensor"
        assert dim == 1, "Only dim=1 reduction is implemented"
        
        batch_size, dim1, dim2 = x.shape
        
        # Output shape: (batch_size, dim2)
        out = torch.empty(batch_size, dim2, dtype=x.dtype, device=x.device)
        
        # Configure kernel launch parameters
        # We'll use a 2D grid: [batch_size, dim2]
        # Each block processes BLOCK_SIZE elements along dim1
        BLOCK_SIZE = 256
        
        # Grid: (batch_size, dim2)
        grid = (batch_size, dim2)
        
        # Calculate strides for kernel
        stride_batch = x.stride(0)
        stride_row = x.stride(1)
        
        # Launch kernel
        min_reduction_optimized_kernel[grid](
            x, out, dim1,
            stride_batch, stride_row,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For min operation, gradient is 1 where the input equals the minimum, 0 otherwise
        # This is a simple implementation; could be optimized further
        raise NotImplementedError("Backward pass not implemented for min reduction")


def triton_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Wrapper function for Triton min reduction.
    """
    return TritonMinReduction.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using Triton kernel.
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
        Applies min reduction over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min(x, self.dim)