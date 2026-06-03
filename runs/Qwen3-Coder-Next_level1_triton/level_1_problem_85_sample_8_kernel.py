import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,              # Input tensor pointer (B, C, H, W)
    w_ptr,              # Weight tensor pointer (C, 1, KH, KW)
    b_ptr,              # Bias tensor pointer (C,) - optional
    out_ptr,            # Output tensor pointer (B, C, OH, OW)
    B, C, H, W,         # Input dimensions
    KH, KW,             # Kernel dimensions
    stride_h, stride_w, # Stride
    pad_h, pad_w,       # Padding
    dil_h, dil_w,       # Dilation
    OH, OW,             # Output dimensions
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    bid = tl.program_id(0)  # Batch index
    cid = tl.program_id(1)  # Channel index
    oh = tl.program_id(2)   # Output height index
    
    # Calculate starting positions for this program
    x_ptr += bid * (H * W * C) + cid * (H * W)
    out_ptr += bid * (OH * OW * C) + cid * (OH * OW)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    
    # Handle bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + cid)
        acc += bias
    
    # Convolution loop over kernel spatial dimensions
    for kh in range(KH):
        h_offset = oh * stride_h + kh * dil_h - pad_h
        # Skip if outside valid input range
        if h_offset < 0 or h_offset >= H:
            continue
            
        for kw in range(KW):
            w_offset = kw  # Will be computed per column
            # Load kernel weight
            w = tl.load(w_ptr + cid * (KH * KW) + kh * KW + kw)
            
            # Process output width dimension with tiling
            for ow_start in range(0, OW, BLOCK_SIZE_W):
                ow_offsets = ow_start + tl.arange(0, BLOCK_SIZE_W)
                mask = ow_offsets < OW
                
                # Calculate input width positions
                in_w = w_offset + ow_offsets * stride_w - pad_w
                
                # Create masks for valid input indices
                w_mask = (in_w >= 0) & (in_w < W)
                valid_mask = w_mask
                
                if tl.sum(valid_mask) == 0:
                    continue
                
                # Load input values
                x_offsets = h_offset * W + in_w
                x_val = tl.load(x_ptr + x_offsets, mask=valid_mask, other=0.0)
                
                # Accumulate convolution result
                acc += tl.where(w_mask, x_val * w, 0.0)
                
            # Store results for this height position
            if kw == KW - 1:  # Only store after processing all kernel width positions
                out_offsets = tl.arange(0, BLOCK_SIZE_H) * OW + oh
                tl.store(out_ptr + out_offsets, acc, mask=tl.arange(0, BLOCK_SIZE_H) < OW)
                acc = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
                if b_ptr is not None:
                    acc += bias

# Optimized version with proper tiling
@triton.jit
def depthwise_conv2d_kernel_optimized(
    x_ptr,              # Input tensor pointer (B, C, H, W)
    w_ptr,              # Weight tensor pointer (C, 1, KH, KW)
    b_ptr,              # Bias tensor pointer (C,) - optional
    out_ptr,            # Output tensor pointer (B, C, OH, OW)
    B, C, H, W,         # Input dimensions
    KH, KW,             # Kernel dimensions
    stride_h, stride_w, # Stride
    pad_h, pad_w,       # Padding
    dil_h, dil_w,       # Dilation
    OH, OW,             # Output dimensions
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    out_h = tl.program_id(2)
    
    # Calculate base pointers for this batch and channel
    x_base = x_ptr + batch_idx * (H * W * C) + channel_idx * (H * W)
    out_base = out_ptr + batch_idx * (OH * OW * C) + channel_idx * (OH * OW)
    
    # Initialize accumulator for the output row
    out_offsets = tl.arange(0, BLOCK_SIZE_W)
    acc = tl.zeros((BLOCK_SIZE_W,), dtype=tl.float32)
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + channel_idx)
        acc += bias
    
    # Convolution over kernel spatial dimensions
    for kh in range(KH):
        in_h = out_h * stride_h + kh * dil_h - pad_h
        
        # Skip if outside valid input range
        if in_h < 0 or in_h >= H:
            continue
            
        for kw in range(KW):
            # Load kernel weight
            w = tl.load(w_ptr + channel_idx * (KH * KW) + kh * KW + kw)
            
            # Calculate input width positions for all output positions
            in_w = out_offsets * stride_w + kw * dil_w - pad_w
            
            # Create masks for valid indices
            valid_mask = (in_w >= 0) & (in_w < W)
            
            # Load input values with masking
            x_offsets = in_h * W + in_w
            x_val = tl.load(x_base + x_offsets, mask=valid_mask, other=0.0)
            
            # Accumulate convolution
            acc += tl.where(valid_mask, x_val * w, 0.0)
    
    # Store results
    out_offsets_final = tl.arange(0, BLOCK_SIZE_W)
    out_ptr_final = out_base + out_offsets_final
    tl.store(out_ptr_final, acc, mask=out_offsets_final < OW)


def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride_h: int = 1,
    stride_w: int = 1,
    padding_h: int = 0,
    padding_w: int = 0,
    dilation_h: int = 1,
    dilation_w: int = 1
) -> torch.Tensor:
    """
    Triton implementation of depthwise 2D convolution.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Weight tensor of shape (C, 1, KH, KW)
        bias: Optional bias tensor of shape (C,)
        stride_h, stride_w: Stride in height and width dimensions
        padding_h, padding_w: Padding in height and width dimensions
        dilation_h, dilation_w: Dilation in height and width dimensions
        
    Returns:
        Output tensor of shape (B, C, OH, OW)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C, H, W = x.shape
    _, _, KH, KW = weight.shape
    
    # Calculate output dimensions
    OH = (H + 2 * padding_h - dilation_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * padding_w - dilation_w * (KW - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, OH, OW), dtype=x.dtype, device=x.device)
    
    # Define block sizes for tiling
    BLOCK_SIZE_C = 1  # Depthwise: one channel per program
    BLOCK_SIZE_H = 1  # One output row per program
    BLOCK_SIZE_W = 64  # Process multiple output width positions per program
    
    # Grid configuration
    grid = (B, C, OH)
    
    # Launch kernel
    depthwise_conv2d_kernel_optimized[grid](
        x, weight, bias, out,
        B, C, H, W,
        KH, KW,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        OH, OW,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_KH=1,
        BLOCK_SIZE_KW=1,
        num_warps=4,
        num_stages=3
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        
        # Create weight and bias parameters
        # Note: The original implementation uses in_channels for both input and output channels
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 2D convolution using Triton kernel.
        """
        return triton_depthwise_conv2d(
            x, self.weight, self.bias,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w
        )