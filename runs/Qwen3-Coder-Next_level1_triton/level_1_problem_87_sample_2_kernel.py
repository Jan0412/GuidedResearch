import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pointwise_conv_kernel(
    x_ptr,  # Input tensor (B, C_in, H, W)
    w_ptr,  # Weight tensor (C_out, C_in)
    b_ptr,  # Bias tensor (C_out,) or None
    out_ptr,  # Output tensor (B, C_out, H, W)
    batch_size,  # B
    in_channels,  # C_in
    out_channels,  # C_out
    height,  # H
    width,  # W
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Batch and spatial index
    batch_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    w_idx = tl.program_id(0)
    
    # Each program handles one output position (batch, h, w) and computes all output channels
    # We'll compute in blocks of output channels
    
    # Calculate the linear index for this (batch, h, w) position
    pos = batch_idx * (height * width) + h_idx * width + w_idx
    
    # Pointers to the input value at this position (same for all output channels)
    x_offset = pos * in_channels
    
    # We'll compute output channels in chunks
    # Each block computes BLOCK_SIZE_N output channels
    
    # Accumulator for the current position
    output = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    # Matrix multiplication: [1, C_in] @ [C_in, BLOCK_SIZE_N] = [1, BLOCK_SIZE_N]
    # Since input is 1xC_in and weights are C_in x BLOCK_SIZE_N
    
    # Iterate over K dimension (in_channels) in blocks
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Load input block: [1, BLOCK_SIZE_K]
        x_offsets = x_offset + tl.arange(0, BLOCK_SIZE_K)
        x_mask = x_offsets < (pos + 1) * in_channels
        x_block = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        
        # Load weight block: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        k_start = k
        n_start = tl.program_id(0) * BLOCK_SIZE_N  # This is actually the output channel offset
        
        # We need to adjust for the fact that program_id(0) gives us the w_idx, not n_start
        # Let's restructure: each program should handle one output channel
        
    # Actually, let's use a better grid layout:
    # Grid: (out_channels, height, batch_size) - each program handles one output channel for one spatial position
    
    # Redefine the kernel with better grid structure
    pass


@triton.jit
def pointwise_conv_kernel_v2(
    x_ptr,  # Input tensor (B, C_in, H, W) - row-major
    w_ptr,  # Weight tensor (C_out, C_in) - row-major
    b_ptr,  # Bias tensor (C_out,) or None
    out_ptr,  # Output tensor (B, C_out, H, W)
    batch_size,  # B
    in_channels,  # C_in
    out_channels,  # C_out
    height,  # H
    width,  # W
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Grid: (out_channels * height * batch_size,)
    # Each program handles one output position: (batch, h, w) for one output channel
    
    # Get the global program ID
    pid = tl.program_id(0)
    
    # Calculate batch, h, w from pid
    # We want: pid = batch_idx * (height * width * out_channels) + h_idx * (width * out_channels) + w_idx * out_channels + out_c_idx
    # But let's use: pid = out_c_idx * (batch_size * height * width) + batch_idx * (height * width) + h_idx * width + w_idx
    # So: out_c_idx = pid / (batch_size * height * width)
    #     remainder = pid % (batch_size * height * width)
    #     batch_idx = remainder / (height * width)
    #     remainder = remainder % (height * width)
    #     h_idx = remainder / width
    #     w_idx = remainder % width
    
    spatial_size = height * width
    batch_spatial_size = batch_size * spatial_size
    
    out_c_idx = pid // batch_spatial_size
    remainder = pid % batch_spatial_size
    
    batch_idx = remainder // spatial_size
    remainder = remainder % spatial_size
    
    h_idx = remainder // width
    w_idx = remainder % width
    
    # Calculate the input pointer offset for this (batch, h, w)
    pos = batch_idx * spatial_size + h_idx * width + w_idx
    x_offset = pos * in_channels
    
    # Calculate output pointer offset
    out_pos = batch_idx * (out_channels * spatial_size) + h_idx * width * out_channels + w_idx * out_channels + out_c_idx
    out_offset = out_pos
    
    # Matrix multiplication: dot product of input row and weight row
    # x: [1, C_in], w[out_c_idx]: [C_in]
    
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over in_channels in blocks
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Load input block
        k_start = k
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < in_channels
        x_block = tl.load(x_ptr + x_offset + k_offsets, mask=k_mask, other=0.0)
        
        # Load weight block
        w_offsets = out_c_idx * in_channels + k_offsets
        w_block = tl.load(w_ptr + w_offsets, mask=k_mask, other=0.0)
        
        # Accumulate
        acc += tl.sum(x_block * w_block)
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_idx)
        acc += bias
    
    # Store result
    tl.store(out_ptr + out_offset, acc.to(tl.float32))


def triton_pointwise_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
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
    
    # Configure kernel parameters
    spatial_size = height * width
    total_outputs = batch_size * out_channels * spatial_size
    
    # Tunable parameters
    BLOCK_SIZE_N = 1  # Not used in this kernel, but kept for interface consistency
    BLOCK_SIZE_K = 256  # Block size for in_channels
    
    # Determine grid size
    grid = (total_outputs,)
    
    # Launch kernel
    pointwise_conv_kernel_v2[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, height, width,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the pointwise 2D convolution using custom Triton kernels.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep the same layer structure for compatibility
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_flag = bias
        
        # Register weight and bias as parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized pointwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return triton_pointwise_conv(x, self.weight, self.bias)