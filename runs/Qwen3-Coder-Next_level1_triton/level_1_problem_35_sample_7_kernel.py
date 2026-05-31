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
    Mean,  # pointer to mean (optional, for inference)
    Rstd,  # pointer to inverse std (optional, for inference)
    N,  # number of elements in a group
    C,  # number of channels
    G,  # number of groups
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program id
    pid = tl.program_id(0)
    
    # Calculate which group this program handles
    group_id = pid
    if group_id >= G:
        return  # Should not happen with correct grid
    
    # Compute start and end channel indices for this group
    start_c = group_id * (C // G)
    end_c = start_c + (C // G)
    
    # Loop over batch and spatial dimensions
    for b in range(tl.cdiv(tl.load(X + 0), 1)):  # dummy load to get batch size
        break
    batch_size = tl.load(X + 0)  # This is just for type inference; we'll compute it properly below
    spatial_size = tl.load(X + 0)  # Dummy
    # Instead, we'll use a 2D grid: batch_id and spatial_block
    # Let's change grid to (batch_size * spatial_blocks, num_groups)
    # For now, assume we'll launch grid as (num_groups,) and handle batch/spatial in loops
    pass  # We'll implement a better grid below


# Better kernel design: use 2D grid: (num_groups, num_spatial_blocks), and loop over batch
@triton.jit
def group_norm_kernel_2d(
    X,  # pointer to input tensor of shape (B, C, H, W)
    Y,  # pointer to output tensor
    Weight,  # pointer to gamma (C)
    Bias,  # pointer to beta (C)
    B,  # batch size
    C,  # number of channels
    H,  # height
    W,  # width
    G,  # number of groups
    N,  # number of elements per group = C//G * H * W
    Mean_out,  # pointer to mean output (optional, for inference)
    Rstd_out,  # pointer to rstd output (optional, for inference)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # pid0 = group_id, pid1 = spatial_block_id
    group_id = tl.program_id(0)
    hw_block_id = tl.program_id(1)
    
    # Number of channels per group
    channels_per_group = C // G
    start_c = group_id * channels_per_group
    end_c = start_c + channels_per_group
    
    # Spatial size
    hw = H * W
    hw_block_size = BLOCK_SIZE_HW
    hw_start = hw_block_id * hw_block_size
    hw_end = tl.minimum(hw_start + hw_block_size, hw)
    
    # Compute offsets for channels
    c_offsets = start_c + tl.arange(0, channels_per_group)
    
    # Compute mean and variance online
    sum_x = tl.zeros((channels_per_group,), dtype=tl.float32)
    sum_x2 = tl.zeros((channels_per_group,), dtype=tl.float32)
    
    # Iterate over batch and spatial positions
    for b in range(B):
        for hw_idx in range(hw_start, hw_end):
            # Compute linear index
            # hw_idx = h * W + w
            h = hw_idx // W
            w = hw_idx % W
            
            # Compute input pointer offset: b * C * H * W + c * H * W + h * W + w
            base_offset = b * (C * H * W) + h * W + w
            
            # Load x for all channels in the group
            x_ptr = X + base_offset
            x = tl.load(x_ptr + c_offsets * (H * W), mask=c_offsets < C, other=0.0)
            
            # Accumulate sums
            x_f32 = x.to(tl.float32)
            sum_x += x_f32
            sum_x2 += x_f32 * x_f32
    
    # Compute mean and variance
    n_elements = B * hw  # number of elements per channel in the group
    mean = sum_x / n_elements
    var = sum_x2 / n_elements - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # eps = 1e-5 for numerical stability
    
    # Store mean and rstd if requested
    if Mean_out is not None:
        mean_ptr = Mean_out + group_id * channels_per_group
        tl.store(mean_ptr + tl.arange(0, channels_per_group), mean, mask=c_offsets < C)
    if Rstd_out is not None:
        rstd_ptr = Rstd_out + group_id * channels_per_group
        tl.store(rstd_ptr + tl.arange(0, channels_per_group), rstd, mask=c_offsets < C)
    
    # Normalize and apply weight and bias
    for b in range(B):
        for hw_idx in range(hw_start, hw_end):
            h = hw_idx // W
            w = hw_idx % W
            base_offset = b * (C * H * W) + h * W + w
            
            x_ptr = X + base_offset
            y_ptr = Y + base_offset
            
            # Load x for all channels
            x = tl.load(x_ptr + c_offsets * (H * W), mask=c_offsets < C, other=0.0)
            x_f32 = x.to(tl.float32)
            
            # Normalize
            x_norm = (x_f32 - mean) * rstd
            
            # Load weight and bias for this channel
            w_ptr = Weight + c_offsets
            b_ptr = Bias + c_offsets
            weight = tl.load(w_ptr, mask=c_offsets < C, other=0.0)
            bias = tl.load(b_ptr, mask=c_offsets < C, other=0.0)
            
            # Apply affine transform
            y_f32 = x_norm * weight.to(tl.float32) + bias.to(tl.float32)
            
            # Store result
            tl.store(y_ptr + c_offsets * (H * W), y_f32.to(x.dtype), mask=c_offsets < C)


# Optimized kernel: fuse the mean/variance computation and normalization into one pass
@triton.jit
def group_norm_forward_kernel(
    X,  # input tensor (B, C, H, W)
    Y,  # output tensor (B, C, H, W)
    Weight,  # gamma (C)
    Bias,  # beta (C)
    B,  # batch size
    C,  # number of channels
    H,  # height
    W,  # width
    G,  # number of groups
    BLOCK_SIZE_G: tl.constexpr,  # block size for channels per group
    BLOCK_SIZE_HW: tl.constexpr,  # block size for spatial elements
):
    # Grid: (num_groups, ceil(H*W / BLOCK_SIZE_HW))
    group_id = tl.program_id(0)
    hw_block_id = tl.program_id(1)
    
    # Channels per group
    channels_per_group = C // G
    start_c = group_id * channels_per_group
    end_c = start_c + channels_per_group
    
    # Spatial block
    hw = H * W
    hw_start = hw_block_id * BLOCK_SIZE_HW
    hw_end = tl.minimum(hw_start + BLOCK_SIZE_HW, hw)
    
    # Precompute offsets for channels in this group
    c_offsets = tl.arange(0, channels_per_group)
    
    # Accumulate statistics for mean and variance
    sum_x = tl.zeros((channels_per_group,), dtype=tl.float32)
    for b in range(B):
        for hw_idx in range(hw_start, hw_end):
            h = hw_idx // W
            w = hw_idx % W
            offset = b * (C * H * W) + c_offsets * (H * W) + h * W + w
            x = tl.load(X + offset, mask=c_offsets < C, other=0.0)
            sum_x += x.to(tl.float32)
    
    # Compute mean
    n = B * (hw_end - hw_start)
    mean = sum_x / n
    
    # Compute variance
    sum_sq_diff = tl.zeros((channels_per_group,), dtype=tl.float32)
    for b in range(B):
        for hw_idx in range(hw_start, hw_end):
            h = hw_idx // W
            w = hw_idx % W
            offset = b * (C * H * W) + c_offsets * (H * W) + h * W + w
            x = tl.load(X + offset, mask=c_offsets < C, other=0.0)
            diff = x.to(tl.float32) - mean
            sum_sq_diff += diff * diff
    
    var = sum_sq_diff / n
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    
    # Normalize and apply affine transformation
    for b in range(B):
        for hw_idx in range(hw_start, hw_end):
            h = hw_idx // W
            w = hw_idx % W
            offset = b * (C * H * W) + c_offsets * (H * W) + h * W + w
            x = tl.load(X + offset, mask=c_offsets < C, other=0.0)
            
            # Normalize
            x_norm = (x.to(tl.float32) - mean) * rstd
            
            # Load weight and bias
            w_val = tl.load(Weight + c_offsets, mask=c_offsets < C, other=0.0)
            b_val = tl.load(Bias + c_offsets, mask=c_offsets < C, other=0.0)
            
            # Apply affine transform
            y_val = x_norm * w_val + b_val
            
            # Store
            tl.store(Y + offset, y_val.to(X.dtype.element_ty), mask=c_offsets < C)


# Even more optimized: use 3D grid and fuse everything in one kernel pass
@triton.jit
def group_norm_fused_kernel(
    X,  # input tensor (B, C, H, W)
    Y,  # output tensor (B, C, H, W)
    Weight,  # gamma (C)
    Bias,  # beta (C)
    B,  # batch size
    C,  # number of channels
    H,  # height
    W,  # width
    G,  # number of groups
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Grid: (B, G, ceil(H*W / BLOCK_SIZE_HW))
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    hw_block_id = tl.program_id(2)
    
    # Channels per group
    channels_per_group = C // G
    start_c = group_id * channels_per_group
    end_c = start_c + channels_per_group
    
    # Spatial block
    hw = H * W
    hw_start = hw_block_id * BLOCK_SIZE_HW
    hw_end = tl.minimum(hw_start + BLOCK_SIZE_HW, hw)
    
    # Precompute offsets for channels
    c_offsets = tl.arange(0, channels_per_group)
    
    # Compute mean and variance for this group in this batch
    sum_x = tl.zeros((channels_per_group,), dtype=tl.float32)
    for hw_idx in range(hw_start, hw_end):
        h = hw_idx // W
        w = hw_idx % W
        offset = start_c * (H * W) + h * W + w
        x = tl.load(X + batch_id * (C * H * W) + offset, mask=c_offsets < C, other=0.0)
        sum_x += x.to(tl.float32)
    
    n = hw_end - hw_start
    mean = sum_x / n
    
    sum_sq_diff = tl.zeros((channels_per_group,), dtype=tl.float32)
    for hw_idx in range(hw_start, hw_end):
        h = hw_idx // W
        w = hw_idx % W
        offset = start_c * (H * W) + h * W + w
        x = tl.load(X + batch_id * (C * H * W) + offset, mask=c_offsets < C, other=0.0)
        diff = x.to(tl.float32) - mean
        sum_sq_diff += diff * diff
    
    var = sum_sq_diff / n
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    
    # Normalize and apply affine transformation
    for hw_idx in range(hw_start, hw_end):
        h = hw_idx // W
        w = hw_idx % W
        offset = start_c * (H * W) + h * W + w
        
        # Load x for all channels in group
        x = tl.load(X + batch_id * (C * H * W) + offset, mask=c_offsets < C, other=0.0)
        x_norm = (x.to(tl.float32) - mean) * rstd
        
        # Load weight and bias
        w_val = tl.load(Weight + c_offsets, mask=c_offsets < C, other=0.0)
        b_val = tl.load(Bias + c_offsets, mask=c_offsets < C, other=0.0)
        
        # Apply affine
        y_val = x_norm * w_val + b_val
        
        # Store
        tl.store(Y + batch_id * (C * H * W) + offset, y_val.to(X.dtype.element_ty), mask=c_offsets < C)


def group_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Triton implementation of Group Normalization.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Scale tensor of shape (C,)
        bias: Shift tensor of shape (C,)
        num_groups: Number of groups (G)
        eps: Small value for numerical stability
    
    Returns:
        Output tensor of shape (B, C, H, W)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    assert x.is_contiguous(), "Input tensor must be contiguous."
    assert x.dim() == 4, "Input must be 4D (B, C, H, W)."
    
    B, C, H, W = x.shape
    assert C % num_groups == 0, f"Channels ({C}) must be divisible by groups ({num_groups})."
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Triton kernel parameters
    channels_per_group = C // num_groups
    BLOCK_SIZE_HW = 256  # Tune this based on H*W
    
    # Grid dimensions: (batch_size, num_groups, ceil(H*W / BLOCK_SIZE_HW))
    hw = H * W
    grid = (B, num_groups, triton.cdiv(hw, BLOCK_SIZE_HW))
    
    # Launch kernel
    group_norm_fused_kernel[grid](
        x,
        y,
        weight,
        bias,
        B,
        C,
        H,
        W,
        num_groups,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using a custom Triton kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer with Triton optimization.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        # Initialize weight and bias as learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, H, W).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return group_norm_triton(x, self.weight, self.bias, self.num_groups)