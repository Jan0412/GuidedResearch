import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,              # Input tensor: (B, C, H, W)
    w_ptr,              # Weight tensor: (C, K, K)
    b_ptr,              # Bias tensor: (C,) or None
    y_ptr,              # Output tensor: (B, C, H_out, W_out)
    B: tl.constexpr,    # Batch size
    C: tl.constexpr,    # Number of channels (in_channels = out_channels for depthwise)
    H: tl.constexpr,    # Input height
    W: tl.constexpr,    # Input width
    K: tl.constexpr,    # Kernel size
    stride: tl.constexpr,
    padding: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr = 16,
    BLOCK_SIZE_W: tl.constexpr = 16,
    BLOCK_SIZE_K: tl.constexpr = 4,
):
    # Each program processes one output element: (batch, channel, h_out, w_out)
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    block_h = tl.program_id(2)
    block_w = tl.program_id(3)

    # Compute base positions in output
    h_out = block_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_out = block_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Check output bounds
    h_mask = h_out < H_out
    w_mask = w_out < W_out
    hw_mask = h_mask[:, None] & w_mask[None, :]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Compute corresponding input positions
    h_in = h_out * stride - padding
    w_in = w_out * stride - padding

    # Iterate over kernel
    for kh in range(K):
        h_in_k = h_in + kh
        h_in_k_mask = (h_in_k >= 0) & (h_in_k < H)
        
        for kw in range(K):
            w_in_k = w_in + kw
            w_in_k_mask = (w_in_k >= 0) & (w_in_k < W)
            
            # Load input values for this kernel position
            input_offsets = (
                batch_idx * (C * H * W) +
                channel_idx * (H * W) +
                h_in_k[:, None] * W +
                w_in_k[None, :]
            )
            
            # Create combined mask for valid positions
            in_mask = h_in_k_mask[:, None] & w_in_k_mask[None, :]
            valid_mask = in_mask & hw_mask
            
            # Load input, handling out-of-bounds with zeros
            x_val = tl.load(x_ptr + input_offsets, mask=valid_mask, other=0.0)
            
            # Load kernel weight
            w_val = tl.load(w_ptr + channel_idx * K * K + kh * K + kw)
            
            # Accumulate
            acc += x_val * w_val

    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + channel_idx)
        acc += bias

    # Store output
    output_offsets = (
        batch_idx * (C * H_out * W_out) +
        channel_idx * (H_out * W_out) +
        h_out[:, None] * W_out +
        w_out[None, :]
    )
    
    tl.store(y_ptr + output_offsets, acc.to(x_ptr.dtype.element_ty), mask=hw_mask)


class TritonDepthwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Extract dimensions
        B, C, H, W = x.shape
        _, _, K, _ = weight.shape
        
        # Calculate output dimensions
        H_out = (H + 2 * padding - K) // stride + 1
        W_out = (W + 2 * padding - K) // stride + 1
        
        # Allocate output tensor
        y = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Define grid dimensions
        # (batch, channel, block_h, block_w)
        grid = lambda meta: (
            B,
            C,
            (H_out + meta['BLOCK_SIZE_H'] - 1) // meta['BLOCK_SIZE_H'],
            (W_out + meta['BLOCK_SIZE_W'] - 1) // meta['BLOCK_SIZE_W']
        )
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, weight, bias, y,
            B, C, H, W, K,
            stride, padding, H_out, W_out,
            BLOCK_SIZE_H=8,
            BLOCK_SIZE_W=8,
            BLOCK_SIZE_K=4
        )
        
        # Save for backward pass (not needed for inference-only, but kept for completeness)
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.H_out = H_out
        ctx.W_out = W_out
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # Not implemented for simplicity - using PyTorch's autograd
        raise NotImplementedError


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0):
    return TritonDepthwiseConv2d.apply(x, weight, bias, stride, padding)


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernel.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register the weight and bias as parameters
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_buffer('bias', None)
            
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)