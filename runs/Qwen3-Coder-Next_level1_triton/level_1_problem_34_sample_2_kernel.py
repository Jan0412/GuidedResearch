import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    X,  # Input tensor pointer [B, C, H, W]
    Y,  # Output tensor pointer [B, C, H, W]
    mean_ptr,  # Mean tensor pointer [B, C]
    rstd_ptr,  # Reciprocal standard deviation tensor pointer [B, C]
    weight_ptr,  # Weight tensor pointer [C]
    bias_ptr,  # Bias tensor pointer [C]
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Compute mean and variance
    # Load the slice of data for this (batch, channel)
    elem_idx = tl.arange(0, BLOCK_SIZE)
    
    # Total elements per (batch, channel) = H * W
    total_elems = H * W
    
    # Compute offsets for this (batch, channel) across all spatial positions
    # Offset formula: batch_idx * stride_b + channel_idx * stride_c + h * stride_h + w * stride_w
    # We'll process in blocks for memory efficiency
    
    # Accumulators for mean and variance
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over spatial dimensions in chunks
    num_blocks = (total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for i in range(num_blocks):
        start = i * BLOCK_SIZE
        offsets = start + elem_idx
        mask = offsets < total_elems
        
        # Compute h, w from linear offset
        h = offsets // W
        w = offsets % W
        
        # Compute memory offset
        offset_ptr = batch_idx * stride_b + channel_idx * stride_c + h * stride_h + w * stride_w
        
        # Load data
        x = tl.load(X + offset_ptr, mask=mask, other=0.0).to(tl.float32)
        
        # Accumulate for mean and variance
        sum_val = tl.where(mask, sum_val + x, sum_val)
        sum_sq_val = tl.where(mask, sum_sq_val + x * x, sum_sq_val)
    
    # Reduce across the BLOCK_SIZE dimension (since each thread holds partial sums)
    sum_val = tl.sum(sum_val, axis=0)
    sum_sq_val = tl.sum(sum_sq_val, axis=0)
    
    # Final reduction across blocks (we need to do this in two steps)
    # But for simplicity, let's assume BLOCK_SIZE >= H*W or we do proper reduction
    # Actually, better approach: compute mean and variance directly
    
    # For correctness, we need to sum across all spatial elements
    # Use a more efficient approach: store partial results in shared memory and reduce
    
    # Alternative: compute mean and variance in a single pass with proper reduction
    
    # Let's do a cleaner implementation using atomic operations or shared memory
    # For simplicity, we'll compute mean and variance in two passes with shared memory
    
    # Actually, Triton provides tl.sum, but we need to be careful with the data layout
    # Let's use a simpler approach with proper memory access pattern
    
    # Redo: compute mean and variance per (batch, channel) using shared memory
    # We'll use a different strategy: load all spatial elements into shared memory
    
    # But since H*W can be large, we'll use a loop and accumulate in float32
    # Since we already did this, let's continue with the calculation
    
    # Compute mean
    mean = sum_val / total_elems
    
    # Compute variance = E[x^2] - E[x]^2
    var = (sum_sq_val / total_elems) - (mean * mean)
    
    # Compute rstd (reciprocal standard deviation)
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd if pointers are provided (for backward pass in training)
    if mean_ptr is not None:
        mean_ptr[batch_idx * C + channel_idx] = mean
    if rstd_ptr is not None:
        rstd_ptr[batch_idx * C + channel_idx] = rstd
    
    # Now compute the normalized output
    # For each spatial location: y = (x - mean) * rstd * weight + bias
    
    # Load weight and bias for this channel
    w_val = tl.load(weight_ptr + channel_idx) if weight_ptr is not None else 1.0
    b_val = tl.load(bias_ptr + channel_idx) if bias_ptr is not None else 0.0
    
    # Process spatial elements again
    for i in range(num_blocks):
        start = i * BLOCK_SIZE
        offsets = start + elem_idx
        mask = offsets < total_elems
        
        h = offsets // W
        w = offsets % W
        
        offset_ptr = batch_idx * stride_b + channel_idx * stride_c + h * stride_h + w * stride_w
        
        # Load input
        x = tl.load(X + offset_ptr, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize: (x - mean) * rstd * weight + bias
        normalized = (x - mean) * rstd * w_val + b_val
        
        # Store result
        tl.store(Y + offset_ptr, normalized, mask=mask)


class TritonInstanceNorm2d(nn.Module):
    """
    Custom Instance Normalization 2D using Triton kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(TritonInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        y = torch.empty_like(x)
        
        # Compute strides
        stride_b = x.stride(0)
        stride_c = x.stride(1)
        stride_h = x.stride(2)
        stride_w = x.stride(3)
        
        # Allocate mean and rstd tensors if needed for backward (not used in inference)
        # For forward pass only, we don't need to store these
        mean_ptr = None
        rstd_ptr = None
        
        # Grid: one block per (batch, channel) pair
        grid = (B, C)
        
        # Block size: choose a reasonable size based on H*W
        # We'll use 256 as a good default for most cases
        BLOCK_SIZE = min(256, max(1, H * W))
        
        # Launch kernel
        instance_norm_kernel[grid](
            x, y, mean_ptr, rstd_ptr,
            self.weight if self.affine else None,
            self.bias if self.affine else None,
            B, C, H, W,
            stride_b, stride_c, stride_h, stride_w,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return y


class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using Triton kernel.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.inorm = TritonInstanceNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inorm(x)