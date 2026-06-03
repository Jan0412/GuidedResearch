import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def triton_conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, D, H, W)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kD, kH, kW)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, D_out, H_out, W_out)
    # Dimensions
    batch_size, in_channels, out_channels,
    depth, height, width,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    out_d, out_h, out_w,
    # Block sizes
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_C: tl.constexpr,
    BLOCK_KD: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Compute batch and output channel indices
    batch_idx = pid_batch
    out_c_idx = pid_out_c * BLOCK_OUT_C + tl.arange(0, BLOCK_OUT_C)
    out_d_idx = pid_d
    out_h_idx = pid_h
    out_w_idx = pid_w
    
    # Create mask for output channels
    out_c_mask = out_c_idx < out_channels
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_OUT_C,), dtype=tl.float32)
    
    # Loop over input channels
    for in_c in range(in_channels):
        # Compute input indices corresponding to this output position
        # For transposed convolution: input_d = (out_d - pad_d + kD - 1 - dil_d * (kD - 1 - d_k)) // stride_d
        # But it's easier to iterate over kernel positions and compute corresponding input positions
        
        # Iterate over kernel spatial dimensions
        for d_k in range(kD):
            input_d = out_d_idx * stride_d - pad_d + d_k * dil_d
            if input_d >= 0 and input_d < depth:
                # Check if within bounds
                d_valid = (input_d >= 0) & (input_d < depth)
                
                for h_k in range(kH):
                    input_h = out_h_idx * stride_h - pad_h + h_k * dil_h
                    if input_h >= 0 and input_h < height:
                        h_valid = (input_h >= 0) & (input_h < height)
                        
                        for w_k in range(kW):
                            input_w = out_w_idx * stride_w - pad_w + w_k * dil_w
                            if input_w >= 0 and input_w < width:
                                w_valid = (input_w >= 0) & (input_w < width)
                                
                                # Load input value
                                input_idx = (
                                    batch_idx * (in_channels * depth * height * width) +
                                    in_c * (depth * height * width) +
                                    input_d * (height * width) +
                                    input_h * width +
                                    input_w
                                )
                                x_val = tl.load(x_ptr + input_idx, mask=d_valid & h_valid & w_valid, other=0.0)
                                
                                # Load weight value
                                weight_idx = (
                                    in_c * (out_channels * kD * kH * kW) +
                                    out_c_idx * (kD * kH * kW) +
                                    d_k * (kH * kW) +
                                    h_k * kW +
                                    w_k
                                )
                                w_val = tl.load(w_ptr + weight_idx, mask=out_c_mask & d_valid & h_valid & w_valid, other=0.0)
                                
                                # Accumulate
                                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_idx = out_c_idx
        bias_val = tl.load(b_ptr + bias_idx, mask=out_c_mask, other=0.0)
        acc += bias_val
    
    # Store result
    out_idx = (
        batch_idx * (out_channels * out_d * out_h * out_w) +
        out_c_idx * (out_d * out_h * out_w) +
        out_d_idx * (out_h * out_w) +
        out_h_idx * out_w +
        out_w_idx
    )
    tl.store(out_ptr + out_idx, acc.to(x_ptr.dtype.element_ty), mask=out_c_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """
    Performs 3D transposed convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch, in_channels, D, H, W)
        weight: Weight tensor of shape (in_channels, out_channels, kD, kH, kW)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to input
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor of shape (batch, out_channels, D_out, H_out, W_out)
    """
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kD, kH, kW = weight.shape
    
    # Calculate output dimensions
    out_d = (depth - 1) * stride - 2 * padding + dilation * (kD - 1) + 1
    out_h = (height - 1) * stride - 2 * padding + dilation * (kH - 1) + 1
    out_w = (width - 1) * stride - 2 * padding + dilation * (kW - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    grid = lambda meta: (
        batch_size,
        triton.cdiv(out_channels, meta["BLOCK_OUT_C"]),
        out_d,
        out_h,
        out_w,
    )
    
    # Launch kernel
    triton_conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kD, kH, kW,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        out_d, out_h, out_w,
        BLOCK_BATCH=1,
        BLOCK_OUT_C=8,
        BLOCK_KD=3,
        BLOCK_KH=3,
        BLOCK_KW=3,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for 3D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias similar to nn.ConvTranspose3d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight parameter
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        
        # Create bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Kaiming uniform initialization similar to PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using Triton kernel.
        """
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None
        
        return triton_conv_transpose3d(
            x, weight, bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )