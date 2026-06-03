import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, height, width)
    w_ptr,  # Weight tensor: (out_channels, in_channels, kernel_height, kernel_width)
    out_ptr,  # Output tensor: (batch, out_channels, out_height, out_width)
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get batch and output position indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    
    # Compute output spatial position
    out_h_start = tl.program_id(2) * BLOCK_SIZE_H
    out_w_start = tl.program_id(3) * BLOCK_SIZE_W
    
    # Compute input position corresponding to output position
    in_h_start = out_h_start * stride_h - padding_h
    in_w_start = out_w_start * stride_w - padding_w
    
    # Create spatial offsets
    h_offsets = tl.arange(0, BLOCK_SIZE_H)
    w_offsets = tl.arange(0, BLOCK_SIZE_W)
    h_mask = (out_h_start + h_offsets) < out_h
    w_mask = (out_w_start + w_offsets) < out_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for c_idx in range(0, in_channels, BLOCK_SIZE_C):
        c_end = tl.minimum(c_idx + BLOCK_SIZE_C, in_channels)
        
        # Loop over kernel height
        for kh_idx in range(0, kernel_h, BLOCK_SIZE_KH):
            kh_end = tl.minimum(kh_idx + BLOCK_SIZE_KH, kernel_h)
            
            # Loop over kernel width
            for kw_idx in range(0, kernel_w, BLOCK_SIZE_KW):
                kw_end = tl.minimum(kw_idx + BLOCK_SIZE_KW, kernel_w)
                
                # Compute actual kernel positions
                kh_pos = kh_idx + tl.arange(0, BLOCK_SIZE_KH)
                kw_pos = kw_idx + tl.arange(0, BLOCK_SIZE_KW)
                
                # Load weights: shape (out_c, in_c, kh, kw)
                # We need to index for the specific out_c_idx
                w_ptrs = w_ptr + (
                    out_c_idx * in_channels * kernel_h * kernel_w +
                    tl.arange(0, 1)[:, None] * in_channels * kernel_h * kernel_w +  # out_c offset
                    (c_idx + tl.arange(0, BLOCK_SIZE_C)[None, :]) * kernel_h * kernel_w +  # in_c offset
                    (kh_pos[None, :] + tl.arange(0, BLOCK_SIZE_KH)[:, None] - kh_idx) * kernel_w +  # kh offset
                    (kw_pos[:, None] + tl.arange(0, BLOCK_SIZE_KW)[None, :] - kw_idx)  # kw offset
                ).view(-1)
                
                # Load input: shape (batch, in_c, h, w)
                in_h_pos = in_h_start + kh_pos[None, :] + tl.arange(0, BLOCK_SIZE_KH)[:, None]
                in_w_pos = in_w_start + kw_pos[:, None] + tl.arange(0, BLOCK_SIZE_KW)[None, :]
                
                # Create masks for input positions
                h_pos_mask = (in_h_pos >= 0) & (in_h_pos < in_h)
                w_pos_mask = (in_w_pos >= 0) & (in_w_pos < in_w)
                mask = h_pos_mask & w_pos_mask
                
                # Compute actual input pointers
                x_ptrs = x_ptr + (
                    batch_idx * in_channels * in_h * in_w +
                    (c_idx + tl.arange(0, BLOCK_SIZE_C)[:, None, None]) * in_h * in_w +
                    (in_h_pos + tl.arange(0, BLOCK_SIZE_KH)[:, None, None] - kh_idx) * in_w +
                    (in_w_pos + tl.arange(0, BLOCK_SIZE_KW)[None, :, :] - kw_idx)
                ).view(-1)
                
                # Load data
                x_vals = tl.load(x_ptrs, mask=mask.view(-1), other=0.0).view(BLOCK_SIZE_C, BLOCK_SIZE_KH, BLOCK_SIZE_KW)
                w_vals = tl.load(w_ptrs, mask=mask.view(-1), other=0.0).view(1, BLOCK_SIZE_C, BLOCK_SIZE_KH, BLOCK_SIZE_KW)
                
                # Accumulate convolution
                acc += tl.sum(x_vals[None, :, :, :] * w_vals, axis=1)
    
    # Store result
    out_ptrs = out_ptr + (
        batch_idx * out_channels * out_h * out_w +
        out_c_idx * out_h * out_w +
        (out_h_start + h_offsets)[:, None] * out_w +
        (out_w_start + w_offsets)[None, :]
    )
    tl.store(out_ptrs, acc, mask=(h_mask[:, None] & w_mask[None, :]))


def triton_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Triton-based convolution implementation"""
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, in_channels_group, kernel_h, kernel_w = weight.shape
    
    # Compute output dimensions
    out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Define block sizes for tuning
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 4
    BLOCK_SIZE_C = 16
    BLOCK_SIZE_KH = 11  # kernel_h
    BLOCK_SIZE_KW = 11  # kernel_w
    
    # Grid definition
    grid = (
        batch_size,           # batch dimension
        out_channels,         # output channels
        (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,  # spatial h
        (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W   # spatial w
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
        stride, stride,  # stride_h, stride_w
        padding, padding,  # padding_h, padding_w
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    # Add bias if provided
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        # Use our Triton-based convolution
        return triton_conv2d(x, self.conv1.weight, self.conv1.bias, 
                           stride=self.conv1.stride, 
                           padding=self.conv1.padding, 
                           dilation=self.conv1.dilation, 
                           groups=self.conv1.groups)