import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, height, width)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_height, kernel_width)
    b_ptr,  # Bias tensor: (out_channels,) - optional, can be None
    out_ptr,  # Output tensor: (batch, out_channels, out_height, out_width)
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_N: tl.constexpr,  # Block size for out_channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for computation (in_channels * kernel_h * kernel_w)
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    
    # Compute output coordinates for this program
    out_h_idx = pid_batch * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_w_idx = tl.arange(0, 1)  # We'll process one output width at a time for simplicity
    out_ch_idx = pid_out_ch * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for valid indices
    mask_h = out_h_idx < out_h
    mask_ch = out_ch_idx < out_channels
    
    # Initialize accumulator for bias and conv result
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_ch_idx, mask=mask_ch, other=0.0)
        acc += bias[None, :]
    
    # Convolution computation
    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel height
        for kh in range(kernel_h):
            # Loop over kernel width
            for kw in range(kernel_w):
                # Compute input position
                in_h_idx = out_h_idx * stride_h + kh - pad_h
                in_w_idx = tl.arange(0, 1) * stride_w + kw - pad_w
                
                # Check bounds for input height
                mask_in_h = (in_h_idx >= 0) & (in_h_idx < in_h)
                mask_in_w = (in_w_idx >= 0) & (in_w_idx < in_w)
                mask = mask_h[:, None] & mask_in_w[None, :] & (mask_in_h[:, None])
                
                # Load input values
                x_offsets = (
                    pid_batch * (in_channels * in_h * in_w) +
                    ic * (in_h * in_w) +
                    in_h_idx[:, None] * in_w +
                    in_w_idx[None, :]
                )
                x_val = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
                
                # Load weight values
                w_offsets = (
                    out_ch_idx[:, None] * (in_channels * kernel_h * kernel_w) +
                    ic * (kernel_h * kernel_w) +
                    kh * kernel_w +
                    kw
                )
                w_val = tl.load(w_ptr + w_offsets, mask=mask_ch[:, None], other=0.0)
                
                # Compute convolution contribution
                acc += tl.dot(x_val, w_val, allow_tf32=False)
    
    # Store result
    out_offsets = (
        pid_batch * (out_channels * out_h * out_w) +
        out_ch_idx[:, None] * (out_h * out_w) +
        out_h_idx[None, :] * out_w +
        tl.arange(0, 1)[None, :]
    )
    mask_out = mask_h[None, :] & mask_ch[:, None]
    tl.store(out_ptr + out_offsets, acc, mask=mask_out)


def triton_conv2d(x, weight, bias=None, stride=1, padding=0):
    """Triton-based 2D convolution implementation."""
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_M = 4  # Batch size block
    BLOCK_SIZE_N = 16  # Output channels block
    BLOCK_SIZE_K = 32  # Not really used in this implementation but kept for consistency
    
    # Grid dimensions
    grid = (
        (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias if bias is not None else None, out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
        stride, stride, padding, padding,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        # Replace PyTorch's Conv2d with our Triton implementation
        # Extract parameters from the original conv1 layer
        weight = self.conv1.weight
        bias = self.conv1.bias
        
        # Use triton_conv2d with the same parameters as original conv1
        return triton_conv2d(
            x, weight, bias,
            stride=self.conv1.stride[0],
            padding=self.conv1.padding[0]
        )