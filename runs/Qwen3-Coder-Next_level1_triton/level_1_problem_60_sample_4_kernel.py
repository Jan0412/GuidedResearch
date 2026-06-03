import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer (batch, in_channels, D, H, W)
    w_ptr,  # Weight tensor pointer (out_channels, in_channels, kD, kH, kW)
    b_ptr,  # Bias tensor pointer (out_channels,) - can be None
    out_ptr,  # Output tensor pointer (batch, out_channels, out_D, out_H, out_W)
    # Dimensions
    batch_size, in_channels, out_channels,
    D, H, W,  # Input dimensions
    out_D, out_H, out_W,  # Output dimensions
    kD, kH, kW,  # Kernel dimensions
    stride_d, stride_h, stride_w,  # Strides
    pad_d, pad_h, pad_w,  # Padding
    dil_d, dil_h, dil_w,  # Dilation
    n_elements,  # Total number of output elements
    # Block sizes for tiling
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_CH: tl.constexpr,
    BLOCK_IN_CH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Calculate output position
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate base indices for output
    out_b = pid_batch
    out_c = pid_out_ch
    out_d = pid_d
    out_h = pid_h
    out_w = pid_w
    
    # Check bounds
    if out_b >= batch_size or out_c >= out_channels or out_d >= out_D or out_h >= out_H or out_w >= out_W:
        return
    
    # Compute the starting position in input for this output element
    in_d_start = out_d * stride_d - pad_d
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Accumulator for the convolution
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for in_c in range(in_channels):
        for kd in range(kD):
            in_d = in_d_start + kd * dil_d
            if in_d < 0 or in_d >= D:
                continue
                
            for kh in range(kH):
                in_h = in_h_start + kh * dil_h
                if in_h < 0 or in_h >= H:
                    continue
                    
                for kw in range(kW):
                    in_w = in_w_start + kw * dil_w
                    if in_w < 0 or in_w >= W:
                        continue
                        
                    # Load input value
                    x_offset = ((out_b * in_channels + in_c) * D * H * W + 
                               in_d * H * W + in_h * W + in_w)
                    x_val = tl.load(x_ptr + x_offset)
                    
                    # Load weight value
                    w_offset = ((out_c * in_channels + in_c) * kD * kH * kW + 
                               kd * kH * kW + kh * kW + kw)
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        b_offset = out_c
        bias = tl.load(b_ptr + b_offset)
        acc += bias
    
    # Store result
    out_offset = ((out_b * out_channels + out_c) * out_D * out_H * out_W +
                 out_d * out_H * out_W + out_h * out_W + out_w)
    tl.store(out_ptr + out_offset, acc)


def triton_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride=1, padding=0, dilation=1, groups=1):
    """
    Performs 3D convolution using Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, D, H, W = x.shape
    out_channels, _, kD, kH, kW = weight.shape
    
    # Calculate output dimensions
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_d = pad_h = pad_w = padding
    else:
        pad_d, pad_h, pad_w = padding
        
    if isinstance(dilation, int):
        dil_d = dil_h = dil_w = dilation
    else:
        dil_d, dil_h, dil_w = dilation
    
    out_D = (D + 2 * pad_d - dil_d * (kD - 1) - 1) // stride_d + 1
    out_H = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    out_W = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_D, out_H, out_W, dtype=x.dtype, device=x.device)
    
    # Grid dimensions
    # We'll use 5D grid: (batch, out_channels, out_D, out_H, out_W)
    # For better performance, we'll tile across dimensions
    
    # Define block sizes - these can be tuned for specific hardware
    BLOCK_BATCH = 1
    BLOCK_OUT_CH = 16
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4
    
    # Adjust blocks based on output dimensions
    grid_batch = batch_size
    grid_out_ch = (out_channels + BLOCK_OUT_CH - 1) // BLOCK_OUT_CH
    grid_d = (out_D + BLOCK_D - 1) // BLOCK_D
    grid_h = (out_H + BLOCK_H - 1) // BLOCK_H
    grid_w = (out_W + BLOCK_W - 1) // BLOCK_W
    
    # Launch the kernel with grid that handles blocks
    # But for simplicity and correctness, we'll use a simpler grid first
    # where each thread computes one output element
    
    # Actually, for better performance, let's use a more optimized approach
    # where we process multiple output elements per thread block
    
    # Redefine grid for better performance
    # We'll process multiple output elements in each thread block
    n_elements = batch_size * out_channels * out_D * out_H * out_W
    
    # For this implementation, use a simpler grid where each thread handles one output element
    # but with proper tiling for memory efficiency
    
    # Use 5D grid for direct computation
    def grid(meta):
        return (batch_size, 
                (out_channels + meta["BLOCK_OUT_CH"] - 1) // meta["BLOCK_OUT_CH"],
                out_D, out_H, out_W)
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        D, H, W, out_D, out_H, out_W,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        n_elements,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT_CH=BLOCK_OUT_CH,
        BLOCK_IN_CH=1,  # Process one input channel at a time for simplicity
        BLOCK_D=1,
        BLOCK_H=1,
        BLOCK_W=1
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton-based 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        # Extract parameters from the original conv3d layer
        weight = self.conv3d.weight
        bias = self.conv3d.bias if self.conv3d.bias is not None else None
        stride = self.conv3d.stride
        padding = self.conv3d.padding
        dilation = self.conv3d.dilation
        groups = self.conv3d.groups
        
        # For simplicity in this implementation, we'll handle groups=1 case
        # and use the Triton kernel. For groups > 1, we'd need to modify the kernel.
        if groups != 1:
            # Fall back to PyTorch for grouped convolutions
            return F.conv3d(x, weight, bias, stride, padding, dilation, groups)
        
        # Use Triton kernel for standard convolutions
        return triton_conv3d(x, weight, bias, stride, padding, dilation, groups)