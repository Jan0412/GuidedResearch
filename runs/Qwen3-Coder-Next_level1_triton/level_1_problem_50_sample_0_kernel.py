import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel for 2D convolution
@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, height, width)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_h, kernel_w)
    b_ptr,  # Bias tensor: (out_channels,)
    out_ptr,  # Output tensor: (batch, out_channels, out_h, out_w)
    batch_size, in_channels, out_channels,
    in_h, in_w,
    out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for computation
):
    # Get program IDs
    pid_m = tl.program_id(0)  # For output channels
    pid_n = tl.program_id(1)  # For batch
    
    # Compute output channel indices
    out_c_start = pid_m * BLOCK_SIZE_M
    out_c_offsets = out_c_start + tl.arange(0, BLOCK_SIZE_M)
    out_c_mask = out_c_offsets < out_channels
    
    # Compute batch index
    batch_idx = pid_n
    
    # Output spatial indices
    for oh in range(out_h):
        for ow in range(out_w):
            # Compute input spatial indices
            in_h_start = oh * stride_h - pad_h
            in_w_start = ow * stride_w - pad_w
            
            # Accumulate convolution result
            acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            
            # Add bias if provided
            if b_ptr is not None:
                b = tl.load(b_ptr + out_c_offsets, mask=out_c_mask, other=0.0)
                acc += b.to(tl.float32)
            
            # Compute convolution
            for ic in range(in_channels):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        # Compute input position
                        h_pos = in_h_start + kh
                        w_pos = in_w_start + kw
                        
                        # Check bounds
                        in_bounds = (h_pos >= 0) & (h_pos < in_h) & (w_pos >= 0) & (w_pos < in_w)
                        
                        if in_bounds:
                            # Compute input pointer offset
                            in_offset = batch_idx * (in_channels * in_h * in_w) + \
                                       ic * (in_h * in_w) + \
                                       h_pos * in_w + w_pos
                            x_val = tl.load(x_ptr + in_offset)
                            
                            # Compute weight pointer offset
                            w_offset = (out_c_offsets * (in_channels * kernel_h * kernel_w) +
                                       ic * (kernel_h * kernel_w) +
                                       kh * kernel_w + kw)
                            w_val = tl.load(w_ptr + w_offset, mask=out_c_mask, other=0.0)
                            
                            acc += x_val * w_val.to(tl.float32)
            
            # Store result
            out_offset = (batch_idx * (out_channels * out_h * out_w) +
                         out_c_offsets * (out_h * out_w) +
                         oh * out_w + ow)
            tl.store(out_ptr + out_offset, acc, mask=out_c_mask)


def triton_conv2d(x, weight, bias=None, stride=1, padding=0):
    """
    Triton-based 2D convolution implementation.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Compute output dimensions
    out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Configure block sizes
    BLOCK_SIZE_M = 16  # For output channels
    BLOCK_SIZE_N = 4   # For batch
    BLOCK_SIZE_K = 32  # Not used in this implementation but kept for flexibility
    
    # Grid configuration
    grid = lambda meta: (
        triton.cdiv(out_channels, meta["BLOCK_SIZE_M"]),
        batch_size
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_h, in_w,
        out_h, out_w,
        kernel_h, kernel_w,
        stride, stride,  # stride_h, stride_w
        padding, padding,  # pad_h, pad_w
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # Initialize weights from original conv1
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()
    
    def forward(self, x):
        # Replace PyTorch's Conv2d with our Triton implementation
        return triton_conv2d(x, self.conv1.weight, self.conv1.bias, 
                            stride=self.conv1.stride, 
                            padding=self.conv1.padding)