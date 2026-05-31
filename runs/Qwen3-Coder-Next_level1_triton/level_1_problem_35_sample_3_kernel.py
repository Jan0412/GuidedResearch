import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def groupnorm_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    Weight,  # pointer to the weight
    Bias,  # pointer to the bias
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    batch_size,  # batch size
    num_groups,  # number of groups
    C,  # number of channels
    H,  # height
    W,  # width
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one group in one batch
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    
    # Calculate the start index for this batch and group
    # Each group has C//num_groups channels
    group_channels = C // num_groups
    start_channel = group_id * group_channels
    
    # Compute the total number of elements in this group (spatial dimensions * channels per group)
    N = H * W * group_channels
    
    # Compute mean
    sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        # We need to handle the case where channels are not contiguous
        # For each element in the block, we need to compute its actual index
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        # Calculate the actual position in the tensor
        # Input layout: (batch, C, H, W)
        # For a given flattened index within the group: idx = (channel_idx - start_channel) * (H*W) + spatial_idx
        # But we're flattening the group: spatial_idx = h * W + w, channel_offset = channel_idx - start_channel
        # So: channel_idx = start_channel + channel_offset, and flattened = channel_idx * (H*W) + spatial_idx
        # = (start_channel + channel_offset) * (H*W) + spatial_idx
        
        # Simplified: we'll iterate over spatial positions and channels
        # But for simplicity, let's compute each element's index explicitly
        
        # Actually, let's change approach: iterate over all elements in the group
        # For each element in the group, compute its flattened index
        # But this is complex. Let's use a different strategy.
        
        # For now, use a simpler approach: we'll compute mean and variance in two passes
        # First pass: compute mean
        pass
    
    # Let's rewrite with a cleaner approach
    # We'll compute mean and variance in separate loops for clarity
    
    # First pass: compute mean
    acc_sum = 0.0
    for i in range(N):
        # Calculate the actual index in the tensor
        # For a given flattened index within the group (0 to N-1)
        # We need to map it to the actual (batch, channel, h, w) indices
        # flattened_group_idx = i
        # channel_offset = i // (H*W)  # which channel in the group
        # spatial_idx = i % (H*W)     # which spatial position
        # h = spatial_idx // W
        # w = spatial_idx % W
        
        channel_offset = i // (H * W)
        spatial_idx = i % (H * W)
        h = spatial_idx // W
        w = spatial_idx % W
        
        actual_channel = start_channel + channel_offset
        
        # Compute the index in the flattened tensor
        # X layout: (batch_size, C, H, W)
        # flattened index = batch_id * (C*H*W) + actual_channel * (H*W) + h * W + w
        idx = batch_id * (C * H * W) + actual_channel * (H * W) + h * W + w
        
        x_val = tl.load(X + idx).to(tl.float32)
        acc_sum += x_val
    
    mean = acc_sum / N
    
    # Second pass: compute variance
    acc_var_sum = 0.0
    for i in range(N):
        channel_offset = i // (H * W)
        spatial_idx = i % (H * W)
        h = spatial_idx // W
        w = spatial_idx % W
        
        actual_channel = start_channel + channel_offset
        idx = batch_id * (C * H * W) + actual_channel * (H * W) + h * W + w
        
        x_val = tl.load(X + idx).to(tl.float32)
        diff = x_val - mean
        acc_var_sum += diff * diff
    
    var = acc_var_sum / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd for backward pass (if needed) or reuse
    # For inference, we might not need to store them, but for correctness we'll store
    if group_id == 0 and batch_id == 0:
        # We only need to store one mean/rstd per group per batch, but we'll store them in a compact way
        pass
    
    # Third pass: normalize and apply weight/bias
    for i in range(N):
        channel_offset = i // (H * W)
        spatial_idx = i % (H * W)
        h = spatial_idx // W
        w = spatial_idx % W
        
        actual_channel = start_channel + channel_offset
        idx = batch_id * (C * H * W) + actual_channel * (H * W) + h * W + w
        
        x_val = tl.load(X + idx).to(tl.float32)
        # Normalize
        normalized = (x_val - mean) * rstd
        # Apply weight and bias
        w_val = tl.load(Weight + actual_channel).to(tl.float32)
        b_val = tl.load(Bias + actual_channel).to(tl.float32)
        y_val = normalized * w_val + b_val
        
        tl.store(Y + idx, y_val.to(Y.dtype.element_ty))


def groupnorm_triton(x, weight, bias, num_groups, eps):
    batch_size, C, H, W = x.shape
    assert C % num_groups == 0, "Number of channels must be divisible by number of groups"
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Grid: one block per (batch, group) combination
    grid = (batch_size, num_groups)
    
    # Launch kernel with appropriate block size
    # Since we're doing sequential computation per group, we don't need large blocks
    # But we'll use a reasonable block size for the inner loops
    BLOCK_SIZE = 256
    
    # Note: The above kernel is simplified but inefficient for large N
    # Let's implement a more efficient version with proper tiling
    
    # Actually, let's rewrite with a better approach that uses shared memory and parallelism
    
    # For now, use the above kernel but with a more efficient implementation below
    groupnorm_kernel[grid](x, y, weight, bias, None, None, batch_size, num_groups, C, H, W, eps, BLOCK_SIZE)
    
    return y


# Better implementation with better parallelization
@triton.jit
def groupnorm_kernel_faster(
    X,  # pointer to the input, shape (batch_size, C, H, W)
    Y,  # pointer to the output
    Weight,  # pointer to the weight, shape (C,)
    Bias,  # pointer to the bias, shape (C,)
    batch_size, 
    num_groups, 
    C, 
    H, 
    W, 
    eps,
):
    # Each program handles one group in one batch
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    
    group_channels = C // num_groups
    start_channel = group_id * group_channels
    
    # Compute mean using parallel reduction
    # We'll use a simple approach with shared memory
    
    # Initialize accumulators
    sum = 0.0
    sum_sq = 0.0
    
    # Iterate over all elements in this group
    N = group_channels * H * W
    
    # Use tiling for better performance
    TILE_SIZE = 128
    for start_n in range(0, N, TILE_SIZE):
        offsets_n = start_n + tl.arange(0, TILE_SIZE)
        mask_n = offsets_n < N
        
        # Calculate indices for these offsets
        channel_offsets = offsets_n // (H * W)
        spatial_indices = offsets_n % (H * W)
        h_indices = spatial_indices // W
        w_indices = spatial_indices % W
        
        actual_channels = start_channel + channel_offsets
        idx = batch_id * (C * H * W) + actual_channels * (H * W) + h_indices * W + w_indices
        
        # Load values
        x_vals = tl.load(X + idx, mask=mask_n, other=0.0).to(tl.float32)
        
        # Accumulate
        sum += tl.sum(x_vals)
        sum_sq += tl.sum(x_vals * x_vals)
    
    # Compute mean and variance
    mean = sum / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Now normalize and apply weight/bias
    for start_n in range(0, N, TILE_SIZE):
        offsets_n = start_n + tl.arange(0, TILE_SIZE)
        mask_n = offsets_n < N
        
        channel_offsets = offsets_n // (H * W)
        spatial_indices = offsets_n % (H * W)
        h_indices = spatial_indices // W
        w_indices = spatial_indices % W
        
        actual_channels = start_channel + channel_offsets
        idx = batch_id * (C * H * W) + actual_channels * (H * W) + h_indices * W + w_indices
        
        x_vals = tl.load(X + idx, mask=mask_n, other=0.0).to(tl.float32)
        
        # Normalize
        normalized = (x_vals - mean) * rstd
        
        # Load weight and bias for these channels
        w_vals = tl.load(Weight + actual_channels, mask=mask_n, other=0.0).to(tl.float32)
        b_vals = tl.load(Bias + actual_channels, mask=mask_n, other=0.0).to(tl.float32)
        
        # Apply transformation
        y_vals = normalized * w_vals + b_vals
        
        tl.store(Y + idx, y_vals.to(Y.dtype.element_ty), mask=mask_n)


def groupnorm_triton_fast(x, weight, bias, num_groups, eps=1e-5):
    batch_size, C, H, W = x.shape
    assert C % num_groups == 0, "Number of channels must be divisible by number of groups"
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Grid: one block per (batch, group) combination
    grid = (batch_size, num_groups)
    
    # Launch kernel
    groupnorm_kernel_faster[grid](x, y, weight, bias, batch_size, num_groups, C, H, W, eps)
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using custom Triton kernels.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer with Triton kernel optimization.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        # Initialize weight and bias as in nn.GroupNorm
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5  # Default epsilon for numerical stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return groupnorm_triton_fast(x, self.weight, self.bias, self.num_groups, self.eps)