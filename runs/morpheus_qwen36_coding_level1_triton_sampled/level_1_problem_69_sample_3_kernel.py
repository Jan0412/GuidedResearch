import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,          # Input tensor pointer [B, C_in, H_in, W_in]
    w_ptr,          # Weight tensor pointer [C_out, C_in, K_H, K_W]
    b_ptr,          # Bias pointer [C_out]
    y_ptr,          # Output tensor pointer [B, C_out, H_out, W_out]
    B, C_in, H_in, W_in,
    C_out, K_H, K_W, H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    out_pad_h, out_pad_w,
    dil_h, dil_w,
    groups,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Grid mapping: each program handles a tile of (B, C_out, H, W)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate global output coordinates
    off_b = pid_b
    off_c = pid_c
    off_h = pid_h * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    off_w = pid_w * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    
    # Mask for valid output coordinates
    mask_h = off_h < H_out
    mask_w = off_w < W_out
    mask_hw = mask_h[:, None] & mask_w[None, :]
    
    # Load bias for this output channel
    bias_val = tl.load(b_ptr + off_c) if b_ptr is not None else 0.0
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_HW, BLOCK_SIZE_HW), dtype=tl.float32)
    
    # Determine input channel range based on groups
    group_id = off_c // (C_out // groups)
    c_in_start = group_id * (C_in // groups)
    c_in_end = (group_id + 1) * (C_in // groups)
    
    # Iterate over input channels
    for c_in in tl.range(c_in_start, c_in_end, BLOCK_SIZE_C):
        # Load input channel pointer offset
        x_offset = (off_b * C_in + c_in) * H_in * W_in
        x_ptr_c = x_ptr + x_offset
        
        # Iterate over kernel height
        for kh in tl.range(K_H):
            # Iterate over kernel width
            for kw in tl.range(K_W):
                # Calculate input coordinates
                # h_in = h_out - kh * stride + pad
                # w_in = w_out - kw * stride + pad
                h_in = off_h - kh * stride_h + pad_h
                w_in = off_w - kw * stride_w + pad_w
                
                # Apply dilation
                h_in = h_in * dil_h
                w_in = w_in * dil_w
                
                # Create mask for valid input coordinates
                mask_in_h = (h_in >= 0) & (h_in < H_in)
                mask_in_w = (w_in >= 0) & (w_in < W_in)
                mask_in = mask_in_h[:, None] & mask_in_w[None, :]
                
                # Load input values
                x_vals = tl.load(x_ptr_c + h_in[:, None] * W_in + w_in[None, :], mask=mask_in, other=0.0)
                
                # Load weight values
                w_ptr_c = w_ptr + (off_c * C_in + c_in) * K_H * K_W
                w_vals = tl.load(w_ptr_c + kh * K_W + kw, mask=mask_in, other=0.0)
                
                # Accumulate
                acc += x_vals * w_vals
    
    # Add bias and store output
    acc += bias_val
    
    # Store result
    y_offset = (off_b * C_out + off_c) * H_out * W_out
    tl.store(y_ptr + y_offset + off_h[:, None] * W_out + off_w[None, :], acc, mask=mask_hw)


def triton_conv_transpose2d(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    dilation: tuple,
    groups: int
) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()
    
    B, C_in, H_in, W_in = x.shape
    C_out, _, K_H, K_W = w.shape
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_H - 1) + output_padding[0] + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_W - 1) + output_padding[1] + 1
    
    y = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Grid configuration
    BLOCK_SIZE_C = 8
    BLOCK_SIZE_HW = 4
    
    grid = (
        B,
        C_out,
        (H_out + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW,
        (W_out + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    )
    
    conv_transpose2d_kernel[grid](
        x, w, b, y,
        B, C_in, H_in, W_in,
        C_out, K_H, K_W, H_out, W_out,
        stride[0], stride[1],
        padding[0], padding[1],
        output_padding[0], output_padding[1],
        dilation[0], dilation[1],
        groups,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized transposed 2D convolution using custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups
        )