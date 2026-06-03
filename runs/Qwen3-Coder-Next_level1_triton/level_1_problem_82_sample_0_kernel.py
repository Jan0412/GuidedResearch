import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor: (B, C, H, W)
    w_ptr,  # Weight tensor: (C, K, K)
    b_ptr,  # Bias tensor: (C,) or None
    out_ptr,  # Output tensor: (B, C, H_out, W_out)
    B, C, H, W, K,  # Dimensions
    stride, padding,  # Convolution parameters
    H_out, W_out,  # Output spatial dimensions
    BLOCK_H: tl.constexpr,  # Block size for height
    BLOCK_W: tl.constexpr,  # Block size for width
    BLOCK_K: tl.constexpr,  # Kernel size block (usually K)
):
    # Program IDs for batch, channel, and spatial position
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    block_h = tl.program_id(2)
    block_w = tl.program_id(3)
    
    # Compute output spatial position
    out_h = block_h * BLOCK_H
    out_w = block_w * BLOCK_W
    
    # Create spatial offsets for output
    h_offsets = tl.arange(0, BLOCK_H)
    w_offsets = tl.arange(0, BLOCK_W)
    out_h_indices = out_h + h_offsets
    out_w_indices = out_w + w_offsets
    
    # Create mask for valid output positions
    h_mask = out_h_indices < H_out
    w_mask = out_w_indices < W_out
    mask = h_mask[:, None] & w_mask[None, :]
    
    # Calculate input starting position
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Kernel indices
    k_h_offsets = tl.arange(0, BLOCK_K)
    k_w_offsets = tl.arange(0, BLOCK_K)
    
    # Process kernel
    for kh in range(K):
        for kw in range(K):
            # Input positions for this kernel element
            in_h = in_h_start + kh
            in_w = in_w_start + kw
            
            # Check bounds for input
            h_valid = (in_h >= 0) & (in_h < H)
            w_valid = (in_w >= 0) & (in_w < W)
            
            if h_valid and w_valid:
                # Calculate input pointer offset
                input_offset = batch_idx * C * H * W + channel_idx * H * W + in_h * W + in_w
                
                # Load input value (scalar for this kernel position)
                x_val = tl.load(x_ptr + input_offset)
                
                # Load weight value
                weight_offset = channel_idx * K * K + kh * K + kw
                w_val = tl.load(w_ptr + weight_offset)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + channel_idx)
        acc += bias
    
    # Store result
    out_offset = batch_idx * C * H_out * W_out + channel_idx * H_out * W_out + out_h * W_out + out_w
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_depthwise_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0):
    """
    Triton implementation of depthwise 2D convolution.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Weight tensor of shape (C, K, K)
        bias: Optional bias tensor of shape (C,)
        stride: Convolution stride
        padding: Padding applied to input
    
    Returns:
        Output tensor of shape (B, C, H_out, W_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C, H, W = x.shape
    _, K, _ = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Configure grid: (batch, channel, H_blocks, W_blocks)
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_K = 3  # For typical kernel sizes
    
    grid = (B, C, (H_out + BLOCK_H - 1) // BLOCK_H, (W_out + BLOCK_W - 1) // BLOCK_W)
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        B, C, H, W, K,
        stride, padding,
        H_out, W_out,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_K=BLOCK_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias similar to nn.Conv2d
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create weight parameter with proper initialization
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size, kernel_size))
        
        # Create bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(x, self.weight, self.bias, 
                                      self.stride, self.padding)


import math