import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr, 
    B, C, H_in, W_in, 
    H_out, W_out, 
    K, S, P, 
    has_bias,
    BLOCK_H: tl.constexpr, 
    BLOCK_W: tl.constexpr,
):
    # Parallelize over (batch * channel), height_blocks, and width_blocks
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Decompose pid_bc into batch and channel
    b = pid_bc // C
    c = pid_bc % C

    # Calculate spatial offsets for the output block
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # Output mask to handle boundary conditions
    out_mask = (h_offsets[:, None] < H_out) & (w_offsets[None, :] < W_out)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Convolution loop over the kernel window
    for kh in range(K):
        for kw in range(K):
            # Load weight for the current channel and kernel position
            # Weight shape: (C, 1, K, K)
            w_val = tl.load(weight_ptr + c * K * K + kh * K + kw)

            # Calculate input coordinates
            h_idx = h_offsets[:, None] * S + kh - P
            w_idx = w_offsets[None, :] * S + kw - P

            # Input mask for padding
            in_mask = (h_idx >= 0) & (h_idx < H_in) & (w_idx >= 0) & (w_idx < W_in)
            
            # Calculate input pointer offset
            # Input shape: (B, C, H_in, W_in)
            x_offset = (b * C + c) * H_in * W_in + h_idx * W_in + w_idx
            x_val = tl.load(x_ptr + x_offset, mask=in_mask, other=0.0)

            acc += x_val * w_val

    # Add bias if applicable
    if has_bias:
        bias_val = tl.load(bias_ptr + c)
        acc += bias_val

    # Store the result
    # Output shape: (B, C, H_out, W_out)
    out_offset = (b * C + c) * H_out * W_out + h_offsets[:, None] * W_out + w_offsets[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def triton_depthwise_conv2d(x, weight, bias, stride, padding):
    # Input shapes
    B, C, H_in, W_in = x.shape
    _, _, K, _ = weight.shape
    S = stride
    P = padding

    # Calculate output dimensions
    H_out = (H_in + 2 * P - K) // S + 1
    W_out = (W_in + 2 * P - K) // S + 1

    # Prepare output tensor
    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Block sizes for spatial dimensions
    BLOCK_H = 16
    BLOCK_W = 16

    # Grid: (Batch * Channels, H_out_blocks, W_out_blocks)
    grid = (B * C, (H_out + BLOCK_H - 1) // BLOCK_H, (W_out + BLOCK_W - 1) // BLOCK_W)

    depthwise_conv2d_kernel[grid](
        x, weight, bias if bias is not None else 0, out,
        B, C, H_in, W_in,
        H_out, W_out,
        K, S, P,
        1 if bias is not None else 0,
        BLOCK_H=BLOCK_H, 
        BLOCK_W=BLOCK_W
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Maintain the same parameter structure as nn.Conv2d for compatibility
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Parameters are initialized exactly like nn.Conv2d
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights using Kaiming Uniform (mimicking PyTorch Conv2d)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (math.sqrt(fan_in)) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using the Triton kernel.
        """
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding
        )

import math # Required for weight initialization