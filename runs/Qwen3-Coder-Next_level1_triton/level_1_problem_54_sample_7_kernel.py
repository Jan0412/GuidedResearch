import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, D, H, W)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kD, kH, kW)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, out_D, out_H, out_W)
    batch_size, in_channels, out_channels,
    D, H, W,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    out_D, out_H, out_W,
    BLOCK_M: tl.constexpr,  # Batch size dimension
    BLOCK_N: tl.constexpr,  # Output channels dimension
    BLOCK_K: tl.constexpr,  # Accumulator dimension (in_channels * kD * kH * kW)
):
    # Program IDs
    pid_m = tl.program_id(0)  # Batch index
    pid_n = tl.program_id(1)  # Output channel index
    pid_d = tl.program_id(2) // (out_H * out_W)
    pid_h = (tl.program_id(2) // out_W) % out_H
    pid_w = tl.program_id(2) % out_W
    
    # Compute output position
    out_d = pid_d
    out_h = pid_h
    out_w = pid_w
    
    # Compute input starting position
    in_d_start = out_d * stride_d - pad_d
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # Loop over in_channels
    for ic in range(in_channels):
        # Loop over kernel dimensions
        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    # Compute input position
                    in_d = in_d_start + kd * dil_d
                    in_h = in_h_start + kh * dil_h
                    in_w = in_w_start + kw * dil_w
                    
                    # Check bounds
                    valid = (in_d >= 0) & (in_d < D) & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
                    
                    if valid:
                        # Load input value
                        x_offset = pid_m * (in_channels * D * H * W) + \
                                   ic * (D * H * W) + \
                                   in_d * (H * W) + \
                                   in_h * W + \
                                   in_w
                        x_val = tl.load(x_ptr + x_offset)
                        
                        # Load weight value
                        w_offset = pid_n * (in_channels * kD * kH * kW) + \
                                   ic * (kD * kH * kW) + \
                                   kd * (kH * kW) + \
                                   kh * kW + \
                                   kw
                        w_val = tl.load(w_ptr + w_offset)
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + pid_n)
        acc += bias_val
    
    # Store result
    out_offset = pid_m * (out_channels * out_D * out_H * out_W) + \
                 pid_n * (out_D * out_H * out_W) + \
                 out_d * (out_H * out_W) + \
                 out_h * out_W + \
                 out_w
    tl.store(out_ptr + out_offset, acc.to(tl.float32))


def triton_conv3d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Performs 3D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch, in_channels, D, H, W)
        weight: Weight tensor of shape (out_channels, in_channels, kD, kH, kW)
        bias: Bias tensor of shape (out_channels,) or None
        stride, padding, dilation, groups: Convolution parameters
    
    Returns:
        Output tensor of shape (batch, out_channels, out_D, out_H, out_W)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, D, H, W = x.shape
    out_channels, _, kD, kH, kW = weight.shape
    
    # Calculate output dimensions
    out_D = (D + 2 * padding - dilation * (kD - 1) - 1) // stride + 1
    out_H = (H + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    out_W = (W + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_D, out_H, out_W, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    grid = (batch_size, out_channels, out_D * out_H * out_W)
    
    # Launch kernel with appropriate block sizes
    BLOCK_M = 1
    BLOCK_N = 16  # Process multiple output channels at once
    BLOCK_K = 1   # Process one input channel and kernel element at a time
    
    conv3d_kernel[grid](
        x, weight, bias if bias is not None else None, out,
        batch_size, in_channels, out_channels,
        D, H, W,
        kD, kH, kW,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        out_D, out_H, out_W,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel.
    Optimized with Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        return triton_conv3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )