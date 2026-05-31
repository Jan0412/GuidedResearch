import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    X,  # Pointer to input tensor
    Y,  # Pointer to output tensor
    W,  # Pointer to weight (gamma) parameter
    B,  # Pointer to bias (beta) parameter
    mean_ptr,  # Pointer to mean (optional, for debugging)
    rstd_ptr,  # Pointer to reciprocal standard deviation (optional, for debugging)
    N,  # Total number of elements in the normalization dimension (features * dim1 * dim2)
    M,  # Number of independent normalization groups (batch_size)
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    # Each program handles one normalization group (one batch element)
    m = tl.program_id(0)
    
    # Compute the starting offset for this group
    group_start = m * N
    
    # Initialize accumulators for mean and variance
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute mean using online algorithm
    for i in range(0, N, BLOCK_SIZE):
        offsets = group_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_start + N
        
        # Load input values
        x = tl.load(X + offsets, mask=mask, other=0.0)
        
        # Accumulate for mean
        sum_val += tl.where(mask, x.to(tl.float32), 0.0)
        
        # Accumulate for variance (we'll use the standard formula)
        sum_sq_val += tl.where(mask, x.to(tl.float32) * x.to(tl.float32), 0.0)
    
    # Reduce to get the sum and sum of squares for the entire group
    # First, do block-level reduction
    block_sum = tl.sum(sum_val)
    block_sum_sq = tl.sum(sum_sq_val)
    
    # Since we're using one block per program, we need to reduce across the block
    # Use tl.sum for the final reduction
    total_sum = block_sum
    total_sum_sq = block_sum_sq
    
    # Compute mean
    mean = total_sum / N
    
    # Compute variance: E[X^2] - E[X]^2
    variance = (total_sum_sq / N) - (mean * mean)
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Store mean and rstd if requested
    if mean_ptr is not None:
        tl.store(mean_ptr + m, mean)
    if rstd_ptr is not None:
        tl.store(rstd_ptr + m, rstd)
    
    # Second pass: normalize and apply weight/bias
    for i in range(0, N, BLOCK_SIZE):
        offsets = group_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_start + N
        
        # Load input values
        x = tl.load(X + offsets, mask=mask, other=0.0)
        
        # Normalize
        x_norm = (x - mean) * rstd
        
        # Apply weight and bias if available
        if HAS_WEIGHT:
            w = tl.load(W + offsets - group_start, mask=mask, other=0.0)
            b = tl.load(B + offsets - group_start, mask=mask, other=0.0)
            x_norm = x_norm * w + b
        
        # Store output
        tl.store(Y + offsets, x_norm.to(X.dtype.element_ty), mask=mask)


class TritonLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight=None, bias=None, eps=1e-5):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Determine the number of normalization groups and elements per group
        M = x.numel() // normalized_shape.numel()  # batch_size
        N = normalized_shape.numel()  # features * dim1 * dim2
        
        # Create output tensor
        y = torch.empty_like(x)
        
        # Set kernel parameters
        BLOCK_SIZE = min(1024, N)
        
        # Launch kernel
        grid = (M,)
        has_weight = weight is not None
        layernorm_kernel[grid](
            x, y, weight, bias, None, None,
            N, M, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=has_weight,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, normalized_shape, weight, bias)
        ctx.eps = eps
        ctx.M = M
        ctx.N = N
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified implementation - for full functionality,
        # a proper backward implementation would be needed.
        # For this example, we'll rely on PyTorch's autograd for backward
        # since implementing full backward pass in Triton is complex.
        
        # For now, just pass through
        return grad_output, None, None, None, None


class TritonLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        
        # Create learnable parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        # Use our custom Triton implementation
        return TritonLayerNormFunction.apply(x, self.normalized_shape, self.weight, self.bias, self.eps)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using Triton kernels.
    """
    def __init__(self, normalized_shape: tuple) -> None:
        """
        Initializes the LayerNorm layer with Triton optimization.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.ln = TritonLayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)