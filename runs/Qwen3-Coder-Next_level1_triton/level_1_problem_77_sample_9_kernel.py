import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triton_conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # (batch_size, in_channels, D, H, W)
    w_ptr,  # (in_channels, out_channels, kD, kH, kW)
    b_ptr,  # (out_channels,) or None
    out_ptr,  # (batch_size, out_channels, D_out, H_out, W_out)
    # Dimensions
    batch_size, in_channels, out_channels,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    # Block sizes
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_CH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # output channel group
    pid_d = tl.program_id(2)  # depth block
    pid_h = tl.program_id(3)  # height block
    pid_w = tl.program_id(4)  # width block
    
    # Create ranges for output dimensions
    out_d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    out_h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_c_offsets = pid_c * BLOCK_OUT_CH + tl.arange(0, BLOCK_OUT_CH)
    
    # Create masks for valid indices
    d_mask = out_d_offsets < D_out
    h_mask = out_h_offsets < H_out
    w_mask = out_w_offsets < W_out
    c_mask = out_c_offsets < out_channels
    
    # Create meshgrid for output positions
    out_d, out_h, out_w = tl.meshgrid(out_d_offsets, out_h_offsets, out_w_offsets)
    out_d = out_d.flatten()
    out_h = out_h.flatten()
    out_w = out_w.flatten()
    
    # Calculate corresponding input positions
    in_d = (out_d - pad_d) // stride_d
    in_h = (out_h - pad_h) // stride_h
    in_w = (out_w - pad_w) // stride_w
    
    # Check if the input positions are valid
    valid_in_d = (in_d >= 0) & (in_d < D_in)
    valid_in_h = (in_h >= 0) & (in_h < H_in)
    valid_in_w = (in_w >= 0) & (in_w < W_in)
    valid = valid_in_d & valid_in_h & valid_in_w
    
    # Calculate kernel positions based on dilation
    # For each output position, iterate over kernel positions
    kernel_d_offsets = tl.arange(0, BLOCK_K) if BLOCK_K >= kD else tl.arange(0, kD)
    kernel_h_offsets = tl.arange(0, BLOCK_K) if BLOCK_K >= kH else tl.arange(0, kH)
    kernel_w_offsets = tl.arange(0, BLOCK_K) if BLOCK_K >= kW else tl.arange(0, kW)
    
    # Initialize accumulator
    accum = tl.zeros((BLOCK_D * BLOCK_H * BLOCK_W, BLOCK_OUT_CH), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for ic in range(in_channels):
        # Calculate input pointer offset
        x_batch_offset = pid_b * (in_channels * D_in * H_in * W_in)
        x_ic_offset = ic * (D_in * H_in * W_in)
        x_base_ptr = x_ptr + x_batch_offset + x_ic_offset
        
        # Calculate kernel pointer offset
        w_ic_offset = ic * (out_channels * kD * kH * kW)
        w_base_ptr = w_ptr + w_ic_offset
        
        # For each kernel position
        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    # Calculate input position
                    in_d_pos = (out_d - pad_d - kd * dil_d) // stride_d
                    in_h_pos = (out_h - pad_h - kh * dil_h) // stride_h
                    in_w_pos = (out_w - pad_w - kw * dil_w) // stride_w
                    
                    # Check validity
                    valid_mask = (in_d_pos >= 0) & (in_d_pos < D_in) & \
                                (in_h_pos >= 0) & (in_h_pos < H_in) & \
                                (in_w_pos >= 0) & (in_w_pos < W_in)
                    
                    # Calculate input pointer
                    in_d_offset = in_d_pos * (H_in * W_in)
                    in_h_offset = in_h_pos * W_in
                    x_offset = in_d_offset + in_h_offset + in_w_pos
                    x_ptr_offset = x_base_ptr + x_offset
                    
                    # Calculate kernel pointer
                    w_offset = kd * (out_channels * kH * kW) + \
                              kh * (out_channels * kW) + \
                              kw * out_channels + \
                              out_c_offsets
                    w_ptr_offset = w_base_ptr + w_offset
                    
                    # Load input value
                    x_val = tl.load(x_ptr_offset, mask=valid_mask, other=0.0)
                    x_val = x_val[:, None]  # Shape: (BLOCK_D*BLOCK_H*BLOCK_W, 1)
                    
                    # Load kernel values
                    w_val = tl.load(w_ptr_offset, mask=c_mask, other=0.0)
                    w_val = w_val[None, :]  # Shape: (1, BLOCK_OUT_CH)
                    
                    # Accumulate
                    accum += x_val * w_val * valid_mask[:, None]
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_offsets, mask=c_mask, other=0.0)
        bias = bias[None, :]  # Shape: (1, BLOCK_OUT_CH)
        accum += bias
    
    # Store results
    accum = accum.to(tl.float32)
    
    # Reshape accumulator back to 3D grid
    accum_reshaped = accum.reshape(BLOCK_D, BLOCK_H, BLOCK_W, BLOCK_OUT_CH)
    
    # Calculate output pointer offsets
    out_batch_offset = pid_b * (out_channels * D_out * H_out * W_out)
    out_base_ptr = out_ptr + out_batch_offset
    
    for c in range(BLOCK_OUT_CH):
        if c >= out_channels:
            break
        out_c_ptr = out_base_ptr + c * (D_out * H_out * W_out)
        out_c_offset = out_d * (H_out * W_out) + out_h * W_out + out_w
        out_ptr_offset = out_c_ptr + out_c_offset
        tl.store(out_ptr_offset, accum_reshaped[:, :, :, c].flatten(), mask=(d_mask & h_mask & w_mask))


def triton_conv_transpose3d(x, weight, bias=None, stride=1, padding=0, dilation=1, output_padding=0):
    """
    Performs 3D transposed convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, D, H, W)
        weight: Weight tensor of shape (in_channels, out_channels, kD, kH, kW)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to input
        dilation: Spacing between kernel elements
        output_padding: Additional size added to output
    
    Returns:
        Output tensor of shape (batch_size, out_channels, D_out, H_out, W_out)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, D_in, H_in, W_in = x.shape
    kD, kH, kW = weight.shape[2:]
    out_channels = weight.shape[1]
    
    # Calculate output dimensions
    D_out = (D_in - 1) * stride - 2 * padding + dilation * (kD - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (kH - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (kW - 1) + output_padding + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Configure block sizes
    BLOCK_BATCH = 1
    BLOCK_OUT_CH = 8
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4
    BLOCK_K = 4
    
    # Calculate grid dimensions
    grid = (
        (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH,
        (out_channels + BLOCK_OUT_CH - 1) // BLOCK_OUT_CH,
        (D_out + BLOCK_D - 1) // BLOCK_D,
        (H_out + BLOCK_H - 1) // BLOCK_H,
        (W_out + BLOCK_W - 1) // BLOCK_W,
    )
    
    # Launch kernel
    triton_conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT_CH=BLOCK_OUT_CH,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_K=BLOCK_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton-based 3D transposed convolution.
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
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using Kaiming initialization (similar to PyTorch)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )


import math