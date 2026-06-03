import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def im2col_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    k_ptr,  # Kernel tensor pointer (OC, C, KH, KW)
    bias_ptr,  # Bias tensor pointer (OC,)
    out_ptr,  # Output tensor pointer (N, OC, OH, OW)
    n_elements,  # Total number of output elements
    N, C, H, W,  # Input dimensions
    OC, KH, KW,  # Output channels and kernel dimensions
    OH, OW,  # Output spatial dimensions
    stride_h, stride_w,  # Stride
    padding_h, padding_w,  # Padding
    dilation_h, dilation_w,  # Dilation
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output element index
    out_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_idx < n_elements
    
    # Convert flat output index to (n, oc, oh, ow)
    ow = out_idx % OW
    out_idx //= OW
    oh = out_idx % OH
    out_idx //= OH
    oc = out_idx % OC
    n = out_idx // OC
    
    # Compute starting position in input
    h_start = oh * stride_h - padding_h
    w_start = ow * stride_w - padding_w
    
    # Compute convolution
    acc = 0.0
    if tl.num_programs(0) == 1:  # First block handles bias
        if out_idx < n_elements:
            acc += tl.load(bias_ptr + oc) if bias_ptr is not None else 0.0
    
    # Iterate over input channels and kernel spatial dimensions
    for c in range(C):
        for kh in range(KH):
            for kw in range(KW):
                h = h_start + kh * dilation_h
                w = w_start + kw * dilation_w
                
                # Check if within input bounds
                valid = (h >= 0) & (h < H) & (w >= 0) & (w < W)
                
                # Compute input index
                input_idx = n * (C * H * W) + c * (H * W) + h * W + w
                
                # Load input value
                x_val = tl.load(x_ptr + input_idx, mask=valid & (input_idx < N * C * H * W), other=0.0)
                
                # Load kernel value
                kernel_idx = oc * (C * KH * KW) + c * (KH * KW) + kh * KW + kw
                k_val = tl.load(k_ptr + kernel_idx, mask=kernel_idx < OC * C * KH * KW, other=0.0)
                
                acc += x_val * k_val
    
    # Store result
    tl.store(out_ptr + out_idx, acc, mask=mask)


@triton.jit
def conv2d_kernel_optimized(
    x_ptr,  # Input tensor (N, C, H, W)
    w_ptr,  # Weight tensor (OC, C, KH, KW)
    b_ptr,  # Bias tensor (OC,)
    out_ptr,  # Output tensor (N, OC, OH, OW)
    N, C, H, W,
    OC, KH, KW,
    OH, OW,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Compute output spatial indices
    oh = tl.program_id(0)
    ow = tl.program_id(1)
    oc = tl.program_id(2)
    
    # Compute input starting position
    h_start = oh * stride_h - padding_h
    w_start = ow * stride_w - padding_w
    
    # Accumulator for this output position
    acc = 0.0
    if b_ptr is not None:
        acc += tl.load(b_ptr + oc)
    
    # Iterate over input channels and kernel dimensions
    for c in range(C):
        for kh in range(KH):
            for kw in range(KW):
                h = h_start + kh * dilation_h
                w = w_start + kw * dilation_w
                
                # Check bounds
                valid = (h >= 0) & (h < H) & (w >= 0) & (w < W)
                
                # Compute indices
                input_idx = h * W + w
                kernel_idx = kh * KW + kw
                
                # Load with masks
                x_val = tl.load(
                    x_ptr + c * H * W + input_idx,
                    mask=valid,
                    other=0.0
                )
                w_val = tl.load(
                    w_ptr + oc * (C * KH * KW) + c * (KH * KW) + kernel_idx,
                    mask=True  # Always valid for weights
                )
                
                acc += x_val * w_val
    
    # Store result
    out_idx = oc * (OH * OW) + oh * OW + ow
    tl.store(out_ptr + out_idx, acc)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Triton-based 2D convolution implementation.
    Supports only groups=1 for simplicity (standard convolution).
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    N, C, H, W = x.shape
    OC, _, KH, KW = weight.shape
    
    # Calculate output dimensions
    OH = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)
    
    # Set up kernel parameters
    stride_h = stride_w = stride
    padding_h = padding_w = padding
    dilation_h = dilation_w = dilation
    
    # Use 3D grid: (OH, OW, OC)
    grid = (OH, OW, OC)
    
    # Launch kernel
    conv2d_kernel_optimized[grid](
        x, weight, bias, out,
        N, C, H, W,
        OC, KH, KW,
        OH, OW,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        BLOCK_H=1,
        BLOCK_W=1,
        BLOCK_C=1,
        BLOCK_KH=1,
        BLOCK_KW=1
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize parameters (using same initialization as nn.Conv2d)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


# Import math for initialization
import math