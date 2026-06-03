import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    x_ptr,  # Input: (B, C_in, D_in, H_in, W_in)
    w_ptr,  # Weight: (C_in, C_out // groups, kD, kH, kW)
    b_ptr,  # Bias: (C_out,) or None
    y_ptr,  # Output: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, C_out, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    output_pad_d, output_pad_h, output_pad_w,
    # Block sizes
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output position
    c_out_start = pid_c_out * BLOCK_SIZE_C_out
    d_out = pid_d * BLOCK_SIZE_D
    h_out = pid_h * BLOCK_SIZE_H
    w_out = pid_w * BLOCK_SIZE_W
    
    # Check bounds
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_out)
    mask_c_out = c_out_offsets < C_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C_out,), tl.float32)
    
    # Iterate over input channels
    for c_in in range(0, C_in, BLOCK_SIZE_C_in):
        c_in_offsets = c_in + tl.arange(0, BLOCK_SIZE_C_in)
        mask_c_in = c_in_offsets < C_in
        
        # Process each kernel position
        for kd in range(kD):
            d_in = d_out - kd * stride_d + padding_d - output_pad_d
            if d_in < 0 or d_in >= D_in:
                continue
                
            for kh in range(kH):
                h_in = h_out - kh * stride_h + padding_h - output_pad_h
                if h_in < 0 or h_in >= H_in:
                    continue
                    
                for kw in range(kW):
                    w_in = w_out - kw * stride_w + padding_w - output_pad_w
                    if w_in < 0 or w_in >= W_in:
                        continue
                    
                    # Calculate indices
                    x_idx = pid_b * (C_in * D_in * H_in * W_in) + \
                            c_in_offsets[:, None, None, None] * (D_in * H_in * W_in) + \
                            d_in * (H_in * W_in) + \
                            h_in * W_in + \
                            w_in
                    w_idx = c_in_offsets[:, None, None, None] * (C_out * kD * kH * kW) + \
                            c_out_offsets[None, :, None, None, None] * (kD * kH * kW) + \
                            kd * (kH * kW) + \
                            kh * kW + \
                            kw
                    
                    # Load values
                    x_val = tl.load(x_ptr + x_idx, mask=mask_c_in[:, None, None, None], other=0.0)
                    w_val = tl.load(w_ptr + w_idx, mask=mask_c_in[:, None, None, None, None] & mask_c_out[None, :, None, None, None], other=0.0)
                    
                    # Accumulate
                    acc += tl.sum(x_val * w_val, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + c_out_offsets, mask=mask_c_out, other=0.0)
        acc += b_val
    
    # Store result
    y_idx = pid_b * (C_out * D_out * H_out * W_out) + \
            c_out_offsets[:, None, None, None] * (D_out * H_out * W_out) + \
            d_out * (H_out * W_out) + \
            h_out * W_out + \
            w_out
    
    tl.store(y_ptr + y_idx, acc.to(x_ptr.dtype.element_ty), mask=mask_c_out[:, None, None, None])


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """Wrapper for the triton transposed 3D convolution kernel"""
    B, C_in, D_in, H_in, W_in = x.shape
    C_in_, C_out_, kD, kH, kW = weight.shape
    
    # Calculate output dimensions
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + (kD - 1) + output_padding[0] + 1
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + (kH - 1) + output_padding[1] + 1
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + (kW - 1) + output_padding[2] + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output
    y = torch.empty((B, C_out_, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Block sizes (tunable parameters)
    BLOCK_SIZE_C_out = min(32, C_out_)
    BLOCK_SIZE_C_in = min(16, C_in)
    BLOCK_SIZE_D = 2
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 8
    
    # Grid dimensions
    grid = lambda meta: (
        B,  # batch
        triton.cdiv(C_out_, meta["BLOCK_SIZE_C_out"]),  # output channels
        triton.cdiv(D_out, meta["BLOCK_SIZE_D"]),  # depth
        triton.cdiv(H_out, meta["BLOCK_SIZE_H"]),  # height
        triton.cdiv(W_out, meta["BLOCK_SIZE_W"]),  # width
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out_, groups,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_SIZE_C_out=BLOCK_SIZE_C_out,
        BLOCK_SIZE_C_in=BLOCK_SIZE_C_in,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of the Model using Triton for transposed 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights (same as ConvTranspose3d)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels // groups, *kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights similar to nn.ConvTranspose3d"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding, self.groups
        )