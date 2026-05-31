import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    X,  # Pointer to input tensor
    Y,  # Pointer to output tensor
    W,  # Pointer to weight tensor (gamma)
    B,  # Pointer to bias tensor (beta)
    Mean,  # Pointer to mean array (output)
    Rstd,  # Pointer to reciprocal std array (output)
    N,  # Number of elements per normalization group
    C,  # Total number of groups (batch_size * features)
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one group (one normalization instance)
    group_idx = tl.program_id(0)
    
    # Calculate the offset for the start of this group
    group_start = group_idx * N
    
    # Initialize accumulators for mean and variance
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute partial sums for mean and variance
    for offset in tl.range(0, N, BLOCK_SIZE):
        idx = group_start + offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < (group_start + N)
        x = tl.load(X + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += x
        sum_sq_val += x * x
    
    # Reduce the partial sums to get the total sum and sum of squares
    # Using a simple loop for reduction (could be optimized with tree reduction)
    for i in range(BLOCK_SIZE // 2):
        sum_val = sum_val + tl.roll(sum_val, 1 << i)
        sum_sq_val = sum_sq_val + tl.roll(sum_sq_val, 1 << i)
    
    mean = sum_val[0] / N
    var = sum_sq_val[0] / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and reciprocal std for backward pass (if needed)
    tl.store(Mean + group_idx, mean)
    tl.store(Rstd + group_idx, rstd)
    
    # Normalize and apply weight and bias
    for offset in tl.range(0, N, BLOCK_SIZE):
        idx = group_start + offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < (group_start + N)
        x = tl.load(X + idx, mask=mask, other=0.0)
        x_norm = (x - mean) * rstd
        w = tl.load(W + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        b = tl.load(B + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        out = x_norm * w + b
        tl.store(Y + idx, out, mask=mask)


def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Apply Layer Normalization using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, features, dim1, dim2)
        weight: Gamma parameter of shape (normalized_shape)
        bias: Beta parameter of shape (normalized_shape)
        eps: Epsilon for numerical stability
        
    Returns:
        Output tensor with same shape as x
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get shape information
    *batch_dims, normalized_shape = x.shape[:-1], x.shape[-1]
    
    # Calculate total number of normalization groups (C) and elements per group (N)
    N = x.shape[-1]
    C = x.numel() // N
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Prepare mean and reciprocal std tensors (if needed for backward)
    mean = torch.empty(C, device=x.device, dtype=torch.float32)
    rstd = torch.empty(C, device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = 128  # Tunable parameter
    
    # Create grid: one block per normalization group
    grid = (C,)
    
    # Launch the Triton kernel
    layernorm_kernel[grid](
        x, out, weight, bias, mean, rstd,
        N, C, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using Triton kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with optimized Triton implementation.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        # Create parameters matching the LayerNorm's expected shape
        # Note: We need to handle the case where normalized_shape is a tuple
        # For LayerNorm, the weight and bias should be of the same shape as the last dimension(s)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        # Create parameters with the same shape as the last dimension(s)
        # In this case, normalized_shape is (64, 256, 256), but LayerNorm expects only the last dimension
        # However, looking at the get_inputs() function, x has shape (16, 64, 256, 256)
        # And we want to normalize over the last dimension (256)
        # But the original Model uses nn.LayerNorm(normalized_shape=normalized_shape) where normalized_shape is (64, 256, 256)
        # This would normalize over the last 3 dimensions, but that's not standard LayerNorm behavior.
        
        # Let's correct the understanding: In PyTorch, LayerNorm(normalized_shape) normalizes over the last len(normalized_shape) dimensions
        # So for normalized_shape=(64, 256, 256), it would normalize over the last 3 dimensions: (64, 256, 256)
        # But our input is (16, 64, 256, 256), so it would normalize over dimensions (64, 256, 256) for each of the 16 batches
        
        # Actually, looking more carefully: the normalized_shape parameter defines the shape of the normalized portion
        # For input (16, 64, 256, 256), if normalized_shape=(64, 256, 256), then it normalizes over the last 3 dimensions
        
        # Let's check the input shape: (batch_size, features, dim1, dim2) = (16, 64, 256, 256)
        # And normalized_shape = (features, dim1, dim2) = (64, 256, 256)
        # So we want to normalize over the last 3 dimensions
        
        # For our Triton kernel, we need to handle normalization over multiple dimensions
        # Let's flatten the dimensions to be normalized and process them as a single dimension
        
        # Actually, let's simplify: the example code uses normalized_shape=(features, dim1, dim2)
        # So we'll assume the user wants to normalize over the last 3 dimensions
        
        # For the Triton kernel, we'll compute the mean and variance over the last len(normalized_shape) dimensions
        
        # Create parameters with the correct shape
        # The weight and bias should have shape normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        # Store the normalized_shape for reference
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Get the dimensions to normalize over
        # normalized_shape is the shape of the normalized portion
        # For input of shape (B, C1, C2, C3) and normalized_shape (C1, C2, C3),
        # we normalize over the last 3 dimensions
        
        # Flatten all dimensions except the last len(normalized_shape) dimensions
        *batch_dims, norm_dims = x.shape[:-len(self.normalized_shape)], x.shape[-len(self.normalized_shape):]
        
        # Calculate total number of elements to normalize over
        N = 1
        for d in self.normalized_shape:
            N *= d
        
        # Reshape to (batch_elements, N) for easier processing
        batch_size = 1
        for d in batch_dims:
            batch_size *= d
        
        # Reshape input to (batch_size, N)
        x_flat = x.view(batch_size, N)
        
        # Reshape weight and bias to (N,)
        weight_flat = self.weight.view(N)
        bias_flat = self.bias.view(N)
        
        # Apply our Triton LayerNorm
        out_flat = triton_layernorm(x_flat, weight_flat, bias_flat)
        
        # Reshape back to original shape
        out = out_flat.view(*batch_dims, *norm_dims)
        
        return out