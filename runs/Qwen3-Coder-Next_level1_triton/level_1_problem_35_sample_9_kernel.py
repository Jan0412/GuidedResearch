import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    X,  # pointer to input tensor
    Y,  # pointer to output tensor
    Weight,  # pointer to gamma (scale)
    Bias,  # pointer to beta (shift)
    Mean,  # pointer to mean (for backward pass, but we can compute on the fly for inference)
    Rstd,  # pointer to standard deviation (for backward pass)
    N,  # number of elements per group
    C,  # total number of channels
    G,  # number of groups
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one group (for one sample)
    # Program ID: group_idx * batch_size + sample_idx
    # But let's use a different mapping: batch_idx, group_idx
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    # Compute start channel index for this group
    start_channel = group_idx * (C // G)
    
    # Compute the offset in the input tensor for this (batch, group)
    # Input shape: (batch, C, D1, D2, ...) -> flatten spatial dims
    # For simplicity, assume 4D: (B, C, H, W)
    # Total elements per sample: C * H * W = N_total
    # But we want per-group statistics, so each group has N = (C // G) * H * W elements
    
    # Actually, let's compute N per group dynamically
    # For now, assume N is total spatial size per channel, so per-group size = (C // G) * N_spatial
    # We'll pass N_spatial (spatial size per channel) separately
    pass  # We'll do this in a more flexible way below


# Better kernel: compute mean and variance per group for each sample, then normalize
@triton.jit
def group_norm_inference_kernel(
    X,  # input tensor: (B, C, H, W) or (B, C, *)
    Y,  # output tensor
    Weight,  # gamma: (C,)
    Bias,    # beta: (C,)
    B,       # batch size
    C,       # number of channels
    G,       # number of groups
    N_spatial,  # spatial size per channel (H*W*...)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, group) pair
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    # Number of elements in this group for this sample
    N = (C // G) * N_spatial
    
    # Compute mean and variance online
    sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute start channel and spatial offset
    start_channel = group_idx * (C // G)
    
    # We need to iterate over all elements in this group for this sample
    # Total elements: (C // G) * N_spatial
    group_channel_size = C // G
    
    # Precompute base offsets for the current batch
    # For input tensor X: [B, C, *spatial] -> flatten to [B, C, N_spatial]
    # Let's assume the tensor is contiguous, so we can compute flat index
    
    # For a 4D tensor (B, C, H, W), flat index = ((batch * C + channel) * H + h) * W + w
    # But for generality, we'll assume the tensor is reshaped to [B, C, N_spatial] internally
    
    # We'll use a loop to accumulate sum and sum_sq
    # Since BLOCK_SIZE is for parallelization within a group, we'll use it for spatial dimension
    
    # Actually, better approach: use block size for channel dimension or spatial dimension?
    # Let's use block size for the total number of elements in the group: (C//G)*N_spatial
    
    # Compute global offset for this (batch, group)
    # For tensor [B, C, N_spatial], flat index for sample b, channel c, spatial s:
    #   idx = (b * C + c) * N_spatial + s
    # For group g, channels from g*(C//G) to (g+1)*(C//G)-1
    
    # We'll compute mean and variance in two passes (or one pass with online algorithm)
    
    # Let's do a simple approach: compute sum and sum_sq over the group
    # Use a loop over channels and spatial positions
    
    # But for Triton, better to use vectorized loads and parallel reduction
    # Let's assume BLOCK_SIZE is a divisor of N (group size)
    
    # Compute offsets for this group
    # We'll use a 1D grid over all elements in the group
    # But program_id(0) and program_id(1) already give us batch and group
    
    # Let's restructure: each program handles one group for one sample
    # Then we need to compute N = (C // G) * N_spatial
    
    # For simplicity, let's use a nested loop or a single loop over the group elements
    # We'll use tl.range and tl.load with masks
    
    # Actually, let's use a different grid: one program per group per sample
    # Then we can use tl.arange(0, BLOCK_SIZE) to cover part of the group
    
    # But BLOCK_SIZE needs to be chosen such that it's <= N
    # We'll set BLOCK_SIZE = min(1024, N) or something
    
    # For now, let's assume we can use a single block per group
    # If N is large, we need to use multiple blocks and do reduction
    
    # Let's do a simple version for moderate sizes first
    
    # Compute the start index in the flattened tensor for this (batch, group)
    # Tensor shape: [B, C, N_spatial]
    # For batch b, group g: channels from g*(C//G) to (g+1)*(C//G)-1
    # So start index = (b * C + g * (C // G)) * N_spatial
    start_idx = (batch_idx * C + group_idx * (C // G)) * N_spatial
    
    # Now sum over the group elements
    sum = tl.zeros((1,), dtype=tl.float32)
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over the group elements
    # Since N might be large, use a grid over the group elements
    # But we're already using program_id for batch and group, so let's use tl.arange for part of the group
    
    # For simplicity, let's assume N is small enough to fit in one block, or use a loop
    # We'll use a loop over blocks of size BLOCK_SIZE
    
    # Actually, let's use a simpler approach: compute mean and variance in a single kernel pass
    # with proper reduction
    
    # For now, let's implement the basic version assuming N <= BLOCK_SIZE
    # For larger N, we'd need to use a more complex reduction
    
    # Let's change BLOCK_SIZE to be the total group size, and assume it's reasonable
    
    # Compute offsets within the group
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Compute the actual indices
    indices = start_idx + offsets
    x = tl.load(X + indices, mask=mask, other=0.0)
    x = x.to(tl.float32)
    
    sum = tl.sum(x, axis=0)
    sum_sq = tl.sum(x * x, axis=0)
    
    # Compute mean and variance
    mean = sum / N
    var = sum_sq / N - mean * mean
    rstd = tl.rsqrt(var + eps)
    
    # Now compute the normalized output
    # We need to store mean and rstd for backward pass, but for inference we can skip
    # But let's store them anyway for correctness
    
    # Write mean and rstd
    if tl.program_id(2) == 0:  # One program per (batch, group) writes mean
        tl.store(Mean + batch_idx * G + group_idx, mean)
    if tl.program_id(2) == 1:  # Another program writes rstd
        tl.store(Rstd + batch_idx * G + group_idx, rstd)
    
    # Now compute the output
    # But we need to recompute the normalized values, so let's do it in a second pass
    # Or combine in one kernel by storing intermediate values
    
    # For simplicity, let's do two passes: first compute statistics, then normalize
    pass


# Better approach: two kernels or one kernel with two phases
# Let's implement a fused kernel that computes mean, variance, and normalized output in one go
# But for simplicity and to ensure correctness, let's do:
# 1. Compute mean and variance for each (batch, group)
# 2. Normalize using mean and variance

# Actually, let's implement a single kernel that does everything for inference
@triton.jit
def group_norm_inference_fused_kernel(
    X,  # input tensor: [B, C, N_spatial]
    Y,  # output tensor: [B, C, N_spatial]
    Weight,  # gamma: [C]
    Bias,    # beta: [C]
    B,       # batch size
    C,       # number of channels
    G,       # number of groups
    N_spatial,  # spatial size per channel
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, group) pair
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    
    # Number of elements in this group
    N = (C // G) * N_spatial
    
    # Compute start index for this (batch, group)
    start_idx = (batch_idx * C + group_idx * (C // G)) * N_spatial
    
    # Compute mean and variance online
    sum = tl.zeros((1,), dtype=tl.float32)
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    
    # Use a loop to handle large N
    # For simplicity, assume N <= BLOCK_SIZE for now
    # In practice, we'd use a more sophisticated approach for large N
    
    # Let's use a simple approach with tl.arange and masks
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    indices = start_idx + offsets
    x = tl.load(X + indices, mask=mask, other=0.0)
    x = x.to(tl.float32)
    
    sum = tl.sum(x, axis=0)
    sum_sq = tl.sum(x * x, axis=0)
    
    mean = sum / N
    var = sum_sq / N - mean * mean
    rstd = tl.rsqrt(var + eps)
    
    # Now compute normalized output
    # For each element in the group, apply normalization and scale/shift
    # We need to handle the channel dimension within the group
    
    # Let's restructure: for each channel in the group and each spatial position
    # But it's easier to do in one pass with channel indices
    
    # Compute channel index for each element
    # For element i in [0, N-1], within group:
    #   channel_idx = i // N_spatial  (within [0, C//G-1])
    #   spatial_idx = i % N_spatial
    # But we need global channel index: group_idx * (C // G) + channel_idx
    
    # Let's recompute indices with channel info
    # Actually, let's just normalize and then apply scale/shift
    
    # Normalize: (x - mean) * rstd
    x_norm = (x - mean) * rstd
    
    # Apply scale and shift: y = x_norm * weight[channel] + bias[channel]
    # We need to compute channel index for each element
    # For element i in [0, N-1] in this group:
    #   global_channel = group_idx * (C // G) + (i // N_spatial)
    # But i = offsets, so:
    #   local_channel = offsets // N_spatial
    #   global_channel = group_idx * (C // G) + local_channel
    
    # But offsets // N_spatial might not be correct if we're using a 1D offset
    # Actually, for a flattened [B, C, N_spatial] tensor, within group:
    #   i = local_channel * N_spatial + spatial_idx
    # So local_channel = i // N_spatial, spatial_idx = i % N_spatial
    
    # Compute global channel index
    local_channel = offsets // N_spatial
    global_channel = group_idx * (C // G) + local_channel
    
    # Load weight and bias
    w = tl.load(Weight + global_channel, mask=mask, other=0.0)
    b = tl.load(Bias + global_channel, mask=mask, other=0.0)
    
    # Apply transformation
    y = x_norm * w + b
    
    # Store result
    tl.store(Y + indices, y.to(X.dtype.element_ty), mask=mask)


# For larger N, we need a more sophisticated approach with reduction
# Let's implement a version that handles large tensors
@triton.jit
def group_norm_stats_kernel(
    X,  # input tensor: [B, C, N_spatial]
    Mean,  # output: [B, G]
    Rstd,  # output: [B, G]
    B, C, G, N_spatial,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    
    N = (C // G) * N_spatial
    start_idx = (batch_idx * C + group_idx * (C // G)) * N_spatial
    
    # Accumulate sum and sum_sq
    sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Use a loop to handle N > BLOCK_SIZE
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_id in range(num_blocks):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        indices = start_idx + offsets
        x = tl.load(X + indices, mask=mask, other=0.0)
        x = x.to(tl.float32)
        
        sum += tl.where(mask, x, 0.0)
        sum_sq += tl.where(mask, x * x, 0.0)
    
    # Reduce within the block
    sum = tl.sum(sum, axis=0)
    sum_sq = tl.sum(sum_sq, axis=0)
    
    # Since we're in a single block, this is fine
    # But for multiple blocks, we'd need tl.atomic_add
    
    # For now, assume BLOCK_SIZE >= N, or use a more complex reduction
    # Let's just use a simple approach for now
    
    mean = sum / N
    var = sum_sq / N - mean * mean
    rstd = tl.rsqrt(var + eps)
    
    tl.store(Mean + batch_idx * G + group_idx, mean)
    tl.store(Rstd + batch_idx * G + group_idx, rstd)


@triton.jit
def group_norm_apply_kernel(
    X,  # input: [B, C, N_spatial]
    Y,  # output: [B, C, N_spatial]
    Weight,  # [C]
    Bias,    # [C]
    Mean,    # [B, G]
    Rstd,    # [B, G]
    B, C, G, N_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel, spatial_block)
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    spatial_block = tl.program_id(2)
    
    # Compute group index
    group_idx = channel_idx // (C // G)
    
    # Load mean and rstd for this (batch, group)
    mean = tl.load(Mean + batch_idx * G + group_idx)
    rstd = tl.load(Rstd + batch_idx * G + group_idx)
    
    # Load weight and bias for this channel
    w = tl.load(Weight + channel_idx)
    b = tl.load(Bias + channel_idx)
    
    # Compute spatial offsets for this block
    spatial_start = spatial_block * BLOCK_SIZE
    offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_spatial
    
    # Compute flat index
    idx = (batch_idx * C + channel_idx) * N_spatial + offsets
    x = tl.load(X + idx, mask=mask, other=0.0)
    x = x.to(tl.float32)
    
    # Normalize
    y = (x - mean) * rstd * w + b
    tl.store(Y + idx, y.to(X.dtype.element_ty), mask=mask)


class TritonGroupNorm(nn.Module):
    def __init__(self, num_features, num_groups, eps=1e-5, elementwise_affine=True):
        super(TritonGroupNorm, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # x shape: [B, C, H, W] or [B, C, *]
        original_shape = x.shape
        B, C = x.shape[0], x.shape[1]
        
        # Reshape to [B, C, N_spatial] where N_spatial = prod(spatial_dims)
        N_spatial = 1
        for s in x.shape[2:]:
            N_spatial *= s
        
        x = x.view(B, C, N_spatial)
        
        # Check if we have learnable parameters
        weight = self.weight if self.elementwise_affine else None
        bias = self.bias if self.elementwise_affine else None
        
        # Compute statistics per group
        # Use the two-kernel approach for better performance on large tensors
        
        # Allocate output tensors for statistics
        mean = torch.empty(B, self.num_groups, device=x.device, dtype=torch.float32)
        rstd = torch.empty(B, self.num_groups, device=x.device, dtype=torch.float32)
        
        # Grid for stats kernel: [B, G]
        grid_stats = (B, self.num_groups)
        
        # Use a reasonable BLOCK_SIZE for stats kernel
        # For now, assume N_spatial * (C // G) <= 1024, otherwise use a loop
        N_per_group = (C // self.num_groups) * N_spatial
        BLOCK_SIZE = min(1024, N_per_group)
        
        # Launch stats kernel
        group_norm_stats_kernel[grid_stats](
            x, mean, rstd,
            B, C, self.num_groups, N_spatial,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Allocate output tensor
        y = torch.empty_like(x)
        
        # Grid for apply kernel: [B, C, (N_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE]
        BLOCK_SIZE_apply = 256
        grid_apply = (B, C, (N_spatial + BLOCK_SIZE_apply - 1) // BLOCK_SIZE_apply)
        
        # Launch apply kernel
        if self.elementwise_affine:
            group_norm_apply_kernel[grid_apply](
                x, y, weight, bias, mean, rstd,
                B, C, self.num_groups, N_spatial,
                BLOCK_SIZE=BLOCK_SIZE_apply,
            )
        else:
            # Create dummy weight and bias if not using elementwise_affine
            dummy_weight = torch.ones(C, device=x.device, dtype=x.dtype)
            dummy_bias = torch.zeros(C, device=x.device, dtype=x.dtype)
            group_norm_apply_kernel[grid_apply](
                x, y, dummy_weight, dummy_bias, mean, rstd,
                B, C, self.num_groups, N_spatial,
                BLOCK_SIZE=BLOCK_SIZE_apply,
            )
        
        # Reshape back to original shape
        y = y.view(original_shape)
        return y


class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using Triton kernels.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer with Triton implementation.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.gn = TritonGroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return self.gn(x)