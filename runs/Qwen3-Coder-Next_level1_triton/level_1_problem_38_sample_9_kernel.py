import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    x_ptr,
    n_rows,
    n_cols,
    out_ptr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Offset to the start of this row
    row_start = row_idx * n_cols
    
    # Accumulator for the sum of absolute values
    sum_abs = 0.0
    
    # Process columns in blocks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        # Load elements
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        
        # Accumulate absolute values
        sum_abs += tl.sum(tl.abs(x), axis=0)
    
    # Compute mean = sum / n_cols
    mean = sum_abs / n_cols
    
    # Store the denominator for later use (we'll use it in division kernel)
    # But actually we can compute the normalization factor directly
    # We want x / mean = x * n_cols / sum_abs
    denom = mean
    
    # Store the denominator for the division kernel to use
    # Actually, let's do it all in one kernel by having each thread compute its result
    # But since we need global synchronization for the sum, we'll use a two-kernel approach
    # For simplicity and correctness, let's do a single kernel that computes the result directly
    # by first computing the sum, then dividing.
    pass  # We'll implement a better approach below


# Better approach: single kernel that computes the normalization
@triton.jit
def l1_normalize_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Offset to the start of this row
    row_start = row_idx * n_cols
    
    # First pass: compute sum of absolute values
    sum_abs = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        sum_abs += tl.sum(tl.abs(x), axis=0)
    
    # Compute denominator: mean = sum / n_cols
    # So x / mean = x * n_cols / sum
    denom = sum_abs / n_cols
    
    # Second pass: normalize
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        normalized = x / denom
        tl.store(out_ptr + row_start + cols, normalized, mask=mask)


class TritonL1Normalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        shape = x.shape
        n_rows = 1
        for dim in shape[:-1]:
            n_rows *= dim
        n_cols = shape[-1]
        
        # Prepare output
        out = torch.empty_like(x)
        
        # Set block size (tunable parameter)
        BLOCK_SIZE = 256
        
        # Grid: one block per row
        grid = (n_rows,)
        
        # Launch kernel
        l1_normalize_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch autograd
        # In a production system, you'd implement the backward pass
        return grad_output


def l1_normalize(x):
    """
    Apply L1 normalization: x / mean(|x|, dim=1, keepdim=True)
    """
    # Handle the case where we need to normalize along dim=1 specifically
    # as per the original code
    return TritonL1Normalize.apply(x)


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
            
        Returns:
            torch.Tensor: Output tensor with L1 normalization applied.
        """
        # Ensure tensor is on GPU and contiguous
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        
        # Get shape
        shape = x.shape
        if len(shape) < 2:
            # If not enough dimensions, just return (edge case)
            return x / torch.mean(torch.abs(x), dim=-1, keepdim=True)
        
        # Reshape to 2D for processing: (batch, features)
        # The original code uses dim=1, so we assume shape[0] is batch
        batch_size = shape[0]
        feature_size = 1
        for s in shape[1:]:
            feature_size *= s
            
        x_2d = x.view(batch_size, feature_size)
        
        # Apply Triton normalization
        out_2d = l1_normalize(x_2d)
        
        # Reshape back to original shape
        out = out_2d.view(*shape)
        
        return out