import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def im2col_kernel(
    x_ptr,                # Input tensor pointer (N, C, H, W)
    k_ptr,                # Kernel tensor pointer (OutC, InC, K_h, K_w)
    bias_ptr,             # Bias tensor pointer (OutC,) - optional
    out_ptr,              # Output tensor pointer (N, OutH, OutW, OutC)
    N: tl.constexpr,      # Batch size
    C: tl.constexpr,      # Input channels
    H: tl.constexpr,      # Input height
    W: tl.constexpr,      # Input width
    OutC: tl.constexpr,   # Output channels
    K_h: tl.constexpr,    # Kernel height
    K_w: tl.constexpr,    # Kernel width
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    OutH: tl.constexpr,   # Output height
    OutW: tl.constexpr,   # Output width
    BLOCK_SIZE: tl.constexpr,
):
    # Compute batch, output height, output width indices
    batch_idx = tl.program_id(0)
    out_h = tl.program_id(1)
    out_w = tl.program_id(2)
    
    # Compute input position corresponding to output (out_h, out_w)
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # For each output channel
    for out_c in range(OutC):
        acc = 0.0
        if bias_ptr is not None:
            acc = tl.load(bias_ptr + out_c)
        
        # Compute convolution for this position
        for in_c in range(C):
            for kh in range(K_h):
                for kw in range(K_w):
                    # Compute input coordinates
                    in_h = in_h_start + kh * dil_h
                    in_w = in_w_start + kw * dil_w
                    
                    # Check bounds
                    if (in_h >= 0 and in_h < H and 
                        in_w >= 0 and in_w < W):
                        # Compute indices
                        x_idx = (batch_idx * C * H * W + 
                                in_c * H * W + 
                                in_h * W + in_w)
                        k_idx = (out_c * C * K_h * K_w + 
                                in_c * K_h * K_w + 
                                kh * K_w + kw)
                        
                        x_val = tl.load(x_ptr + x_idx)
                        k_val = tl.load(k_ptr + k_idx)
                        acc += x_val * k_val
        
        # Store result
        out_idx = (batch_idx * OutH * OutW * OutC + 
                  out_h * OutW * OutC + 
                  out_w * OutC + out_c)
        tl.store(out_ptr + out_idx, acc)


# Optimized version using blocked computation for better performance
@triton.jit
def conv2d_kernel_optimized(
    x_ptr,                # Input tensor pointer (N, C, H, W)
    k_ptr,                # Kernel tensor pointer (OutC, C, K_h, K_w)
    bias_ptr,             # Bias tensor pointer (OutC,) - optional
    out_ptr,              # Output tensor pointer (N, OutC, OutH, OutW)
    N: tl.constexpr,      # Batch size
    C: tl.constexpr,      # Input channels
    H: tl.constexpr,      # Input height
    W: tl.constexpr,      # Input width
    OutC: tl.constexpr,   # Output channels
    K_h: tl.constexpr,    # Kernel height
    K_w: tl.constexpr,    # Kernel width
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    OutH: tl.constexpr,   # Output height
    OutW: tl.constexpr,   # Output width
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Compute output position
    batch_idx = tl.program_id(0)
    out_c = tl.program_id(1)
    out_h = tl.program_id(2) * BLOCK_H
    out_w = tl.program_id(3) * BLOCK_W
    
    # Compute input position
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Load bias if available
    bias_val = 0.0
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_c)
    
    # Accumulate convolution result
    acc = tl.full((BLOCK_H, BLOCK_W), bias_val, dtype=tl.float32)
    
    # Process input channels in blocks
    for in_c in range(0, C, BLOCK_C):
        c_range = tl.arange(0, BLOCK_C)
        c_mask = c_range < C - in_c
        in_c_block = in_c + c_range
        
        # Process kernel
        for kh in range(K_h):
            for kw in range(K_w):
                # Compute input coordinates for this kernel position
                in_h = in_h_start + kh * dil_h
                in_w = in_w_start + kw * dil_w
                
                # Load kernel weights
                k_idx = (out_c * C * K_h * K_w + 
                        in_c_block[:, None, None] * K_h * K_w + 
                        kh * K_w + kw)
                k_val = tl.load(k_ptr + k_idx, mask=c_mask[:, None, None], other=0.0)
                
                # Compute valid output region
                out_h_end = min(BLOCK_H, OutH - out_h)
                out_w_end = min(BLOCK_W, OutW - out_w)
                
                if out_h_end > 0 and out_w_end > 0:
                    # Compute input region for this kernel position
                    h_offsets = tl.arange(0, BLOCK_H)
                    w_offsets = tl.arange(0, BLOCK_W)
                    h_mask = (h_offsets < out_h_end)
                    w_mask = (w_offsets < out_w_end)
                    
                    # Compute input indices
                    in_h_offsets = in_h + h_offsets * stride_h
                    in_w_offsets = in_w + w_offsets * stride_w
                    
                    h_valid = (in_h_offsets >= 0) & (in_h_offsets < H)
                    w_valid = (in_w_offsets >= 0) & (in_w_offsets < W)
                    
                    # Load input values
                    x_indices = (batch_idx * C * H * W + 
                                in_c_block[:, None, None] * H * W + 
                                in_h_offsets[None, :, None] * W + 
                                in_w_offsets[None, None, :])
                    
                    x_val = tl.load(x_ptr + x_indices, 
                                  mask=c_mask[:, None, None] & h_valid[None, :, None] & w_valid[None, None, :],
                                  other=0.0)
                    
                    # Multiply and accumulate
                    acc += tl.sum(k_val[None, :, :, :] * x_val[:, :, :, :], axis=1)
    
    # Store results
    h_offsets = tl.arange(0, BLOCK_H)
    w_offsets = tl.arange(0, BLOCK_W)
    h_mask = (h_offsets < min(BLOCK_H, OutH - out_h))
    w_mask = (w_offsets < min(BLOCK_W, OutW - out_w))
    
    out_indices = (batch_idx * OutC * OutH * OutW + 
                  out_c * OutH * OutW + 
                  (out_h + h_offsets[:, None]) * OutW + 
                  (out_w + w_offsets[None, :]))
    
    tl.store(out_ptr + out_indices, acc, mask=h_mask[:, None] & w_mask[None, :])


def triton_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 2D convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "Only groups=1 is supported in this implementation."
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    N, C, H, W = x.shape
    OutC, _, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
        
    if isinstance(dilation, int):
        dil_h = dil_w = dilation
    else:
        dil_h, dil_w = dilation
    
    OutH = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    OutW = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty((N, OutC, OutH, OutW), dtype=x.dtype, device=x.device)
    
    # Set block sizes
    BLOCK_H, BLOCK_W = 4, 4
    BLOCK_C = min(32, C)
    
    # Calculate grid dimensions
    grid = lambda meta: (
        N,
        OutC,
        triton.cdiv(OutH, BLOCK_H),
        triton.cdiv(OutW, BLOCK_W),
    )
    
    # Launch kernel
    conv2d_kernel_optimized[grid](
        x, weight, bias, out,
        N, C, H, W, OutC, K_h, K_w,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        OutH, OutW,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights using kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )