import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    stride_h_in,
    stride_w_in,
    stride_h_out,
    stride_w_out,
    stride_c_in,
    stride_c_out,
    stride_b_in,
    stride_w_in_elem,
    stride_w_out_elem,
    stride_c_in_elem,
    stride_c_out_elem,
    height_in,
    width_in,
    height_out,
    width_out,
    in_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    has_bias,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Grid dimensions
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Block coordinates
    block_h = tl.program_id(2)
    block_w = tl.program_id(3)
    
    # Offsets within block
    off_h = tl.arange(0, BLOCK_H)
    off_w = tl.arange(0, BLOCK_W)
    
    # Global output coordinates
    out_h = block_h * BLOCK_H + off_h
    out_w = block_w * BLOCK_W + off_w
    
    # Mask for valid output elements
    mask_h = out_h < height_out
    mask_w = out_w < width_out
    mask = mask_h[:, None] & mask_w[None, :]
    
    # Input coordinates calculation
    # w_in = stride * out_w + padding
    w_in = stride * out_w + padding
    # h_in depends on kernel offset k
    # We will compute this inside the kernel loop
    
    # Load weights for this channel
    # Weights shape: (in_channels, 1, kernel_size, 1)
    # For channel pid_c, weights are at offset pid_c * kernel_size
    w_ptr_c = w_ptr + pid_c * kernel_size
    weights = tl.load(w_ptr_c + tl.arange(0, kernel_size), mask=tl.arange(0, kernel_size) < kernel_size, other=0.0)
    
    # Load bias if present
    if has_bias:
        b = tl.load(b_ptr + pid_c)
    else:
        b = 0.0
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over kernel height
    for k in tl.static_range(kernel_size):
        # h_in = stride * out_h + padding - dilation * k
        h_in = stride * out_h + padding - dilation * k
        
        # Create mask for valid input coordinates
        mask_h_in = (h_in >= 0) & (h_in < height_in)
        mask_w_in = (w_in >= 0) & (w_in < width_in)
        mask_input = mask_h_in[:, None] & mask_w_in[None, :]
        
        # Load input tile
        # Input strides: (B, C, H, W) -> strides: (C*H*W, H*W, W, 1)
        # We are in block (pid_b, pid_c)
        # Base pointer for this block
        base_ptr = x_ptr + pid_b * stride_c_in + pid_c * stride_c_in_elem
        
        # Offsets for input tile
        # h_in * stride_h_in + w_in * stride_w_in
        # But h_in and w_in are tensors
        # We can compute offsets as:
        # offsets = h_in[:, None] * stride_h_in + w_in[None, :] * stride_w_in
        # However, Triton requires constexpr strides or careful handling
        # Since we have global strides, we can compute:
        # ptr = base_ptr + h_in[:, None] * stride_h_in + w_in[None, :] * stride_w_in
        
        # Load input
        input_tile = tl.load(base_ptr + h_in[:, None] * stride_h_in + w_in[None, :] * stride_w_in, 
                             mask=mask_input, other=0.0)
        
        # Accumulate
        acc += input_tile * weights[k]
    
    # Add bias
    if has_bias:
        acc += b
    
    # Store output
    # Output strides: (B, C, H, W) -> strides: (C*H*W, H*W, W, 1)
    out_base_ptr = out_ptr + pid_b * stride_c_out + pid_c * stride_c_out_elem
    out_offsets = out_h[:, None] * stride_h_out + out_w[None, :] * stride_w_out
    tl.store(out_base_ptr + out_offsets, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out = torch.empty_like(x)
        
        # Calculate output dimensions
        H_out = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        W_out = (W + 2 * self.padding - self.dilation * (1 - 1) - 1) // self.stride + 1
        
        # Strides
        stride_h_in = W
        stride_w_in = 1
        stride_c_in = H * W
        stride_b_in = C * H * W
        
        stride_h_out = W
        stride_w_out = 1
        stride_c_out = H * W
        stride_b_out = C * H * W
        
        # Block sizes
        BLOCK_H = 64
        BLOCK_W = 64
        BLOCK_C = 1
        
        # Grid
        grid = (B, C, (H_out + BLOCK_H - 1) // BLOCK_H, (W_out + BLOCK_W - 1) // BLOCK_W)
        
        # Launch kernel
        depthwise_conv_kernel[grid](
            x, self.conv2d.weight, self.conv2d.bias if self.has_bias else None,
            out,
            stride_h_in, stride_w_in, stride_h_out, stride_w_out,
            stride_c_in, stride_c_out, stride_b_in,
            1, 1, 1, 1,
            H, W, H_out, W_out, C,
            self.kernel_size, self.stride, self.padding, self.dilation,
            self.has_bias,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
        )
        return out


def get_inputs():
    batch_size = 64
    in_channels = 8
    kernel_size = 3
    width = 512
    height = 512
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [8, 3, 1, 0, 1]