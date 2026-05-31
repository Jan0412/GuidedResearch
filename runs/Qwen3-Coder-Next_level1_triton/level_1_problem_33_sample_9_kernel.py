import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def batchnorm_forward_stats_kernel(
    x_ptr,  # Input tensor
    n_elements,  # Total number of elements
    batch_size,  # Batch size
    num_features,  # Number of features
    spatial_size,  # Spatial dimensions product (dim1 * dim2 * ...)
    mean_ptr,  # Output: mean per feature
    var_ptr,   # Output: variance per feature
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
):
    # Each program handles one feature channel
    feature_id = tl.program_id(0)
    
    # Calculate offsets for this feature across all batch and spatial locations
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulators for mean and variance
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=DTYPE)
    sum_sq_val = tl.zeros((BLOCK_SIZE,), dtype=DTYPE)
    
    # Compute total number of elements per feature (batch * spatial)
    total_elements = batch_size * spatial_size
    
    # Process in blocks
    for start in range(0, total_elements, BLOCK_SIZE):
        block_offsets = start + offsets
        mask = block_offsets < total_elements
        
        # Compute global index for this feature
        global_idx = feature_id * total_elements + block_offsets
        x_val = tl.load(x_ptr + global_idx, mask=mask, other=0.0)
        
        sum_val += x_val
        sum_sq_val += x_val * x_val
    
    # Sum across the block dimension
    total_sum = tl.sum(sum_val)
    total_sum_sq = tl.sum(sum_sq_val)
    
    # Compute mean and variance
    mean = total_sum / total_elements
    var = (total_sum_sq / total_elements) - (mean * mean)
    
    # Store results
    tl.store(mean_ptr + feature_id, mean)
    tl.store(var_ptr + feature_id, var)

@triton.jit
def batchnorm_forward_apply_kernel(
    x_ptr,  # Input tensor
    mean_ptr,  # Mean tensor
    var_ptr,   # Variance tensor
    weight_ptr,  # Scale parameter
    bias_ptr,    # Shift parameter
    out_ptr,     # Output tensor
    n_elements,  # Total number of elements
    batch_size,
    num_features,
    spatial_size,
    eps,         # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate feature index for each element
    feature_index = (offsets // spatial_size) % num_features
    
    # Get mean and variance for this feature
    mean = tl.load(mean_ptr + feature_index)
    var = tl.load(var_ptr + feature_index)
    
    # Get weight and bias for this feature
    weight = tl.load(weight_ptr + feature_index)
    bias = tl.load(bias_ptr + feature_index)
    
    # Compute normalization
    std = tl.sqrt(var + eps)
    normalized = (x - mean) / std
    
    # Apply affine transformation
    out = normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

class TritonBatchNorm2d(nn.Module):
    """
    Triton-optimized BatchNorm2d implementation
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics for evaluation
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        
        batch_size, num_features, *spatial_dims = x.shape
        spatial_size = 1
        for dim in spatial_dims:
            spatial_size *= dim
            
        n_elements = x.numel()
        
        # Training mode: compute statistics from current batch
        if self.training:
            # Compute mean and variance using Triton kernel
            mean = torch.empty(num_features, device=x.device, dtype=x.dtype)
            var = torch.empty(num_features, device=x.device, dtype=x.dtype)
            
            BLOCK_SIZE = 128
            grid = (num_features,)
            
            batchnorm_forward_stats_kernel[grid](
                x, n_elements, batch_size, num_features, spatial_size,
                mean, var, BLOCK_SIZE=BLOCK_SIZE, DTYPE=tl.float32
            )
            
            # Update running statistics
            with torch.no_grad():
                # Compute updated running statistics
                factor = 1.0 / (batch_size * spatial_size) if batch_size * spatial_size > 1 else 0
                updated_mean = mean * factor + self.running_mean * (1 - factor)
                updated_var = var * factor + self.running_var * (1 - factor)
                
                self.running_mean.copy_(updated_mean)
                self.running_var.copy_(updated_var)
                
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked.add_(1)
            
            # Apply normalization using Triton kernel
            out = torch.empty_like(x)
            
            # Launch kernel with appropriate grid size
            BLOCK_SIZE = 256
            grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            
            batchnorm_forward_apply_kernel[grid](
                x, mean, var, self.weight, self.bias, out,
                n_elements, batch_size, num_features, spatial_size, self.eps,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            return out
        else:
            # Evaluation mode: use running statistics
            out = torch.empty_like(x)
            
            # Launch kernel with appropriate grid size
            BLOCK_SIZE = 256
            grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            
            batchnorm_forward_apply_kernel[grid](
                x, self.running_mean, self.running_var, self.weight, self.bias, out,
                n_elements, batch_size, num_features, spatial_size, self.eps,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the optimized BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.bn = TritonBatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)