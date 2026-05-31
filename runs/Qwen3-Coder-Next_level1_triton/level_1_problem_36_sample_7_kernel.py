import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,  # Pointer to input tensor
    y_ptr,  # Pointer to output tensor
    weight_ptr,  # Pointer to optional weight (not used in RMSNorm but included for compatibility)
    n_elements,  # Total number of elements
    batch_size,  # Batch size
    num_features,  # Number of features
    dim1,  # First spatial dimension
    dim2,  # Second spatial dimension
    eps,  # Epsilon value
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    
    # Calculate which feature channel this program handles
    feature_idx = pid % num_features
    
    # Calculate the starting offset for this feature in the batch
    # For input shape (batch_size, num_features, dim1, dim2)
    # We process one feature at a time across all batches and spatial locations
    
    # Total elements per feature across batch and spatial dimensions
    elements_per_feature = batch_size * dim1 * dim2
    
    # For each feature, we need to:
    # 1. Compute sum of squares
    # 2. Compute RMS = sqrt(mean + eps)
    # 3. Normalize each element
    
    # Process in blocks
    for block_start in range(0, elements_per_feature, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_feature
        
        # Compute the actual index in the flattened tensor
        # For feature f: indices are f*elements_per_feature + offsets
        base_idx = feature_idx * elements_per_feature
        
        # Load input values for this feature
        x = tl.load(x_ptr + base_idx + offsets, mask=mask, other=0.0)
        
        # Compute squared values
        x_sq = x * x
        
        # For the first block, accumulate sum of squares
        if block_start == 0:
            sum_sq = tl.sum(x_sq)
        else:
            sum_sq += tl.sum(x_sq)
    
    # Compute mean and RMS
    mean_sq = sum_sq / elements_per_feature
    rms = tl.sqrt(mean_sq + eps)
    
    # Now normalize all elements
    for block_start in range(0, elements_per_feature, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_feature
        
        base_idx = feature_idx * elements_per_feature
        x = tl.load(x_ptr + base_idx + offsets, mask=mask, other=0.0)
        
        # Normalize
        y = x / rms
        
        # Store result
        tl.store(y_ptr + base_idx + offsets, y, mask=mask)


# Alternative approach: process entire tensor in a more cache-friendly way
@triton.jit
def rmsnorm_kernel_fused(
    x_ptr,  # Pointer to input tensor (shape: batch_size, num_features, dim1, dim2)
    y_ptr,  # Pointer to output tensor
    batch_size, 
    num_features,
    dim1,
    dim2,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # We'll process one feature at a time for each batch element
    # Grid: (batch_size * dim1 * dim2, num_features)
    
    batch_idx = tl.program_id(0) // (dim1 * dim2)
    spatial_idx = tl.program_id(0) % (dim1 * dim2)
    feature_idx = tl.program_id(1)
    
    # Calculate offsets for this position across all batches
    # For a given (batch, spatial) position, we process all features
    
    # Total elements per (batch, spatial) position across features
    # Actually, we want to compute RMS per feature, so we need to process differently
    
    pass  # This approach won't work directly - we need to compute mean per feature


class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        # x shape: (batch_size, num_features, dim1, dim2)
        batch_size, num_features, dim1, dim2 = x.shape
        
        # Prepare output tensor
        y = torch.empty_like(x)
        
        # Calculate total elements
        n_elements = x.numel()
        
        # For RMSNorm, we need to compute sum of squares along dimension 1 (features)
        # We'll use a different kernel that processes features in parallel
        
        # Kernel launch configuration: one block per feature
        # Each block processes all batch * dim1 * dim2 elements for that feature
        
        # Calculate grid size: one block per feature
        grid = (num_features,)
        BLOCK_SIZE = 1024  # Tunable parameter
        
        # Launch kernel
        _rmsnorm_forward_kernel[grid](
            x, y, 
            batch_size, num_features, dim1, dim2, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return y


@triton.jit
def _rmsnorm_forward_kernel(
    x_ptr, y_ptr,
    batch_size, num_features, dim1, dim2, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one feature across all batch and spatial positions
    feature_idx = tl.program_id(0)
    
    # Total elements per feature
    elements_per_feature = batch_size * dim1 * dim2
    
    # First pass: compute sum of squares
    sum_sq = 0.0
    for block_start in range(0, elements_per_feature, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_feature
        
        # Calculate the actual index in the flattened tensor
        # For feature f: indices are f*elements_per_feature + offsets
        base_idx = feature_idx * elements_per_feature
        flat_idx = base_idx + offsets
        
        # Load input values
        x = tl.load(x_ptr + flat_idx, mask=mask, other=0.0)
        
        # Compute squared values and accumulate
        x_sq = x * x
        sum_sq += tl.sum(x_sq, axis=0, keepdims=False)
    
    # Compute mean and RMS
    mean_sq = sum_sq / elements_per_feature
    rms = tl.sqrt(mean_sq + eps)
    
    # Second pass: normalize and store
    for block_start in range(0, elements_per_feature, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_feature
        
        base_idx = feature_idx * elements_per_feature
        flat_idx = base_idx + offsets
        
        # Load input values
        x = tl.load(x_ptr + flat_idx, mask=mask, other=0.0)
        
        # Normalize
        y = x / rms
        
        # Store result
        tl.store(y_ptr + flat_idx, y, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using Triton kernels.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return TritonRMSNorm.apply(x, self.eps)