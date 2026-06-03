import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor pointer: (C_in, C_out, Kd, Kh, Kw)
    b_ptr,  # Bias tensor pointer: (C_out,) or None
    y_ptr,  # Output tensor pointer: (B, C_out, D_out, H_out, W_out)
    B, C_in, D, H, W,  # Input dimensions
    C_out, Kd, Kh, Kw,  # Weight dimensions
    D_out, H_out, W_out,  # Output dimensions
    stride_d, stride_h, stride_w,  # Stride parameters
    pad_d, pad_h, pad_w,  # Padding parameters
    output_pad_d, output_pad_h, output_pad_w,  # Output padding parameters
    dil_d, dil_h, dil_w,  # Dilation parameters
    C_in_grouped,  # Should be 1 for standard conv, C_in for depthwise
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_Kd: tl.constexpr,
    BLOCK_SIZE_Kh: tl.constexpr,
    BLOCK_SIZE_Kw: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Output indices
    out_d = pid_d
    out_h = pid_h
    out_w = pid_w
    
    # Calculate input indices considering stride, padding, and dilation
    in_d_start = out_d * stride_d - pad_d
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C_out,), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in_offset in range(0, C_in, BLOCK_SIZE_C_in):
        c_in_idx = c_in_offset + tl.arange(0, BLOCK_SIZE_C_in)
        mask_c_in = c_in_idx < C_in
        
        for kd in range(0, Kd, BLOCK_SIZE_Kd):
            for kh in range(0, Kh, BLOCK_SIZE_Kh):
                for kw in range(0, Kw, BLOCK_SIZE_Kw):
                    # Compute kernel indices
                    kd_idx = kd + tl.arange(0, BLOCK_SIZE_Kd)
                    kh_idx = kh + tl.arange(0, BLOCK_SIZE_Kh)
                    kw_idx = kw + tl.arange(0, BLOCK_SIZE_Kw)
                    
                    # Compute input indices
                    in_d = in_d_start + kd_idx * dil_d
                    in_h = in_h_start + kh_idx * dil_h
                    in_w = in_w_start + kw_idx * dil_w
                    
                    # Check bounds for input indices
                    mask_d = (in_d >= 0) & (in_d < D)
                    mask_h = (in_h >= 0) & (in_h < H)
                    mask_w = (in_w >= 0) & (in_w < W)
                    
                    # Create combined mask for input
                    mask_input = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
                    
                    # Load input data
                    x_offsets = (
                        pid_b * (C_in * D * H * W) +
                        c_in_idx[:, None, None, None] * (D * H * W) +
                        in_d[:, None, None, None] * (H * W) +
                        in_h[None, :, None, None] * W +
                        in_w[None, None, :, None]
                    )
                    x_vals = tl.load(
                        x_ptr + x_offsets,
                        mask=mask_input[:, :, :, None] & mask_c_in[:, None, None, None],
                        other=0.0
                    )
                    
                    # Load weight data
                    w_offsets = (
                        c_in_idx[:, None, None, None] * (C_out * Kd * Kh * Kw) +
                        pid_c_out * (Kd * Kh * Kw) +
                        kd_idx[None, :, None, None, None] * (Kh * Kw) +
                        kh_idx[None, None, :, None, None] * Kw +
                        kw_idx[None, None, None, :, None]
                    )
                    w_vals = tl.load(
                        w_ptr + w_offsets,
                        mask=mask_c_in[:, None, None, None] & mask_input[None, :, :, :, :],
                        other=0.0
                    )
                    
                    # Accumulate: x * w
                    acc += tl.sum(x_vals * w_vals, axis=[0, 1, 2, 3])
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Store result
    y_offset = (
        pid_b * (C_out * D_out * H_out * W_out) +
        pid_c_out * (D_out * H_out * W_out) +
        out_d * (H_out * W_out) +
        out_h * W_out +
        out_w
    )
    tl.store(y_ptr + y_offset, acc.to(x_ptr.dtype.element_ty))


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    """
    Triton implementation of ConvTranspose3d forward pass.
    Note: This is a simplified implementation that may not handle all edge cases.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = x.shape
    C_in_weight, C_out, Kd, Kh, Kw = weight.shape
    
    # Handle stride, padding, output_padding, dilation as tuples
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    output_pad_d, output_pad_h, output_pad_w = output_padding
    dil_d, dil_h, dil_w = dilation
    
    # Calculate output dimensions
    D_out = (D - 1) * stride_d - 2 * pad_d + dil_d * (Kd - 1) + output_pad_d + 1
    H_out = (H - 1) * stride_h - 2 * pad_h + dil_h * (Kh - 1) + output_pad_h + 1
    W_out = (W - 1) * stride_w - 2 * pad_w + dil_w * (Kw - 1) + output_pad_w + 1
    
    # Create output tensor
    y = torch.empty((B, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Grid configuration
    grid = (B, C_out, D_out, H_out, W_out)
    
    # Launch kernel with reasonable block sizes
    BLOCK_SIZE_C_out = 16
    BLOCK_SIZE_C_in = 16
    BLOCK_SIZE_Kd = 3
    BLOCK_SIZE_Kh = 3
    BLOCK_SIZE_Kw = 3
    
    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, D, H, W,
        C_out, Kd, Kh, Kw,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        output_pad_d, output_pad_h, output_pad_w,
        dil_d, dil_h, dil_w,
        1,  # C_in_grouped (for standard convolution)
        BLOCK_SIZE_C_out=BLOCK_SIZE_C_out,
        BLOCK_SIZE_C_in=BLOCK_SIZE_C_in,
        BLOCK_SIZE_Kd=BLOCK_SIZE_Kd,
        BLOCK_SIZE_Kh=BLOCK_SIZE_Kh,
        BLOCK_SIZE_Kw=BLOCK_SIZE_Kw,
    )
    
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )