import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batchnorm_forward_kernel(
    x_ptr,  # Input tensor
    running_mean_ptr,  # Running mean (for training mode)
    running_var_ptr,  # Running variance (for training mode)
    weight_ptr,  # Gamma parameter
    bias_ptr,  # Beta parameter
    output_ptr,  # Output tensor
    n_elements,  # Total number of elements in input
    batch_size,  # Batch size
    num_features,  # Number of feature channels
    spatial_size,  # Product of spatial dimensions (dim1 * dim2 * ...)
    eps,  # Epsilon for numerical stability
    M: tl.constexpr,  # Block size for feature dimension
    N: tl.constexpr,  # Block size for spatial dimension
):
    # Get the feature channel index this program instance processes
    feature_idx = tl.program_id(0)
    
    # pointers to current feature's parameters
    weight_ptr += feature_idx
    bias_ptr += feature_idx
    running_mean_ptr += feature_idx
    running_var_ptr += feature_idx
    
    # Load weight and bias for this feature
    weight = tl.load(weight_ptr)
    bias = tl.load(bias_ptr)
    
    # Compute mean and variance across batch and spatial dimensions
    # First pass: compute sum for mean
    sum_val = 0.0
    sum_sq_val = 0.0
    
    # We'll iterate over batch and spatial dimensions in blocks
    for b in range(batch_size):
        for s in range(0, spatial_size, N):
            s_offsets = s + tl.arange(0, N)
            mask = s_offsets < spatial_size
            
            # Calculate global index for this position
            idx = b * num_features * spatial_size + feature_idx * spatial_size + s_offsets
            
            # Load values
            x_vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
            
            # Accumulate sums
            sum_val += tl.sum(x_vals, axis=0)
            sum_sq_val += tl.sum(x_vals * x_vals, axis=0)
    
    # Compute mean and variance
    total_elements = batch_size * spatial_size
    mean = sum_val / total_elements
    var = (sum_sq_val / total_elements) - (mean * mean)
    
    # Update running statistics if in training mode (we assume training for now)
    # In practice, we'd need to handle training/eval mode properly
    # For this example, we'll just use the computed statistics
    
    # Compute standard deviation with epsilon
    std = tl.sqrt(var + eps)
    
    # Second pass: normalize and scale
    for b in range(batch_size):
        for s in range(0, spatial_size, N):
            s_offsets = s + tl.arange(0, N)
            mask = s_offsets < spatial_size
            
            # Calculate global index for this position
            idx = b * num_features * spatial_size + feature_idx * spatial_size + s_offsets
            
            # Load input values
            x_vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
            
            # Normalize
            normalized = (x_vals - mean) / std
            
            # Scale and shift
            out_vals = normalized * weight + bias
            
            # Store output
            tl.store(output_ptr + idx, out_vals, mask=mask)


def triton_batchnorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                     running_mean: torch.Tensor, running_var: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of BatchNorm2d forward pass.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    
    batch_size, num_features, *spatial_dims = x.shape
    spatial_size = 1
    for dim in spatial_dims:
        spatial_size *= dim
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Grid: one block per feature channel
    grid = (num_features,)
    
    # Block sizes for tiling
    BLOCK_SIZE_N = 256  # Spatial dimension block size
    
    # Launch kernel
    batchnorm_forward_kernel[grid](
        x, running_mean, running_var, weight, bias, output,
        x.numel(), batch_size, num_features, spatial_size, eps,
        M=1, N=BLOCK_SIZE_N
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for BatchNorm2d.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer with custom Triton implementation.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        
        # Store parameters for manual computation
        self.num_features = num_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization using custom Triton kernel.
        
        Note: This implementation assumes training mode and uses the current
        batch statistics (not running statistics). For production use, 
        training/eval mode handling would need to be properly implemented.
        """
        # For simplicity, we'll replace the standard BatchNorm with our Triton implementation
        # The_bn parameters are accessible through self.bn
        
        # Get parameters
        weight = self.bn.weight
        bias = self.bn.bias
        
        # Use running statistics from the bn module (initialized with ones/zeros)
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        
        # Call our Triton batchnorm
        return triton_batchnorm(
            x, weight, bias, running_mean, running_var, self.bn.eps
        )