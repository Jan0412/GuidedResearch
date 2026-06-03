import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer (batch, in_channels, height, width)
    w_ptr,  # Weight tensor pointer (out_channels, in_channels)
    b_ptr,  # Bias tensor pointer (out_channels,) - can be None
    out_ptr,  # Output tensor pointer (batch, out_channels, height, width)
    batch_size, in_channels, height, width, out_channels,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output rows (out_channels)
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output columns (batch * height * width)
):
    # Compute program indices
    pid_m = tl.program_id(0)  # For out_channels dimension
    pid_n = tl.program_id(1)  # For batch * height * width dimension
    
    # Create offsets for output dimensions
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Total number of spatial locations (batch * height * width)
    total_spatial = batch_size * height * width
    
    # Masks to ensure we don't go out of bounds
    mask_m = offsets_m < out_channels
    mask_n = offsets_n < total_spatial
    
    # Initialize accumulator for output
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Iterate over in_channels dimension in blocks
    for k in range(0, in_channels, BLOCK_SIZE_N):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_N)
        mask_k = offsets_k < in_channels
        
        # Load input: shape (in_channels, total_spatial)
        # We need to transpose the input for efficient access
        x_block = tl.load(
            x_ptr + offsets_k[:, None] * (height * width) + offsets_n[None, :],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # Load weights: shape (out_channels, in_channels)
        w_block = tl.load(
            w_ptr + offsets_m[:, None] * in_channels + offsets_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Compute partial dot product
        accumulator += tl.sum(w_block * x_block, axis=1)
    
    # Add bias if provided
    if b_ptr is not None:
        bias_offsets = offsets_m
        bias_mask = mask_m
        bias_val = tl.load(b_ptr + bias_offsets, mask=bias_mask, other=0.0)
        accumulator += bias_val
    
    # Store output
    # Output shape: (batch, out_channels, height, width)
    # We need to compute batch, height, width from spatial index
    out_block = accumulator.to(tl.float32)
    
    # For each output channel in the block
    for m in range(BLOCK_SIZE_M):
        if offsets_m[m] < out_channels:
            # Write to all spatial locations
            out_channel_offset = offsets_m[m]
            for n in range(BLOCK_SIZE_N):
                if offsets_n[n] < total_spatial:
                    # Convert spatial index back to batch, height, width
                    spatial_idx = offsets_n[n]
                    batch_idx = spatial_idx // (height * width)
                    remaining = spatial_idx % (height * width)
                    height_idx = remaining // width
                    width_idx = remaining % width
                    
                    out_offset = (
                        batch_idx * (out_channels * height * width) +
                        out_channel_offset * (height * width) +
                        height_idx * width +
                        width_idx
                    )
                    tl.store(out_ptr + out_offset, out_block[m])


def triton_pointwise_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Performs pointwise 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels)
        bias: Optional bias tensor of shape (out_channels,)
    
    Returns:
        Output tensor of shape (batch_size, out_channels, height, width)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, height, width, device=x.device, dtype=x.dtype)
    
    # Total spatial locations
    total_spatial = batch_size * height * width
    
    # Set block sizes (tunable parameters)
    BLOCK_SIZE_M = 32  # For out_channels dimension
    BLOCK_SIZE_N = 128  # For spatial locations
    
    # Grid dimensions
    grid = (
        triton.cdiv(out_channels, BLOCK_SIZE_M),
        triton.cdiv(total_spatial, BLOCK_SIZE_N)
    )
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width, out_channels,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation using optimized Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Use our optimized Triton implementation instead of PyTorch's conv2d
        return triton_pointwise_conv2d(x, self.conv1d.weight, self.conv1d.bias)