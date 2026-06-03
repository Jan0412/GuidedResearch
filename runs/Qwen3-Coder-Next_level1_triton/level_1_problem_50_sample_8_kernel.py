import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor [B, C, H, W]
    w_ptr,  # Weight tensor [out_channels, in_channels, kH, kW]
    out_ptr,  # Output tensor [B, out_channels, out_H, out_W]
    B, C, H, W,
    out_channels, kH, kW,
    stride: tl.constexpr, padding: tl.constexpr, 
    out_H: tl.constexpr, out_W: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for convolution computation
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate output channel and batch indices
    out_c_start = pid_m * BLOCK_SIZE_M
    batch_idx = pid_b * BLOCK_SIZE_N + pid_n
    
    # Create offsets for output channels
    out_c_offsets = out_c_start + tl.arange(0, BLOCK_SIZE_M)
    mask_out_c = out_c_offsets < out_channels
    
    # Create offsets for batch index (we'll handle one batch at a time for simplicity)
    # For simplicity, we'll process one batch at a time and vectorize over channels
    if batch_idx >= B:
        return
    
    # For each output position
    for oh in range(out_H):
        for ow in range(out_W):
            # Calculate input position
            ih = oh * stride - padding
            iw = ow * stride - padding
            
            # Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            
            # Loop over input channels
            for ic in range(C):
                # Loop over kernel height
                for kh in range(kH):
                    # Loop over kernel width
                    for kw in range(kW):
                        # Calculate input position
                        in_h = ih + kh
                        in_w = iw + kw
                        
                        # Check bounds
                        valid_h = (in_h >= 0) & (in_h < H)
                        valid_w = (in_w >= 0) & (in_w < W)
                        valid = valid_h & valid_w
                        
                        # Calculate input pointer offset
                        in_offset = (batch_idx * C * H * W + 
                                    ic * H * W + 
                                    in_h * W + 
                                    in_w)
                        
                        # Load input value
                        x_val = tl.load(x_ptr + in_offset, 
                                       mask=valid, 
                                       other=0.0)
                        
                        # Calculate weight pointer offset
                        w_offset = (out_c_offsets * C * kH * kW + 
                                   ic * kH * kW + 
                                   kh * kW + 
                                   kw)
                        
                        # Load weight values
                        w_val = tl.load(w_ptr + w_offset, 
                                       mask=mask_out_c[:, None], 
                                       other=0.0)
                        w_val = tl.reshape(w_val, (BLOCK_SIZE_M,))
                        
                        # Accumulate
                        acc += x_val * w_val
            
            # Store result
            out_offset = (batch_idx * out_channels * out_H * out_W + 
                         out_c_offsets * out_H * out_W + 
                         oh * out_W + 
                         ow)
            tl.store(out_ptr + out_offset, acc, mask=mask_out_c)


def triton_conv2d(x, weight, bias=None, stride=4, padding=2):
    """
    Triton implementation of Conv2d
    """
    B, C, H, W = x.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(B, out_channels, out_H, out_W, device=x.device, dtype=x.dtype)
    
    # Set up kernel parameters
    BLOCK_SIZE_M = 16  # Output channels per block
    BLOCK_SIZE_N = 4   # Batch per block
    
    # Grid dimensions: (num_blocks_out_channels, num_blocks_batch, batch_size)
    grid = lambda meta: (
        triton.cdiv(out_channels, meta['BLOCK_SIZE_M']),
        min(B, meta['BLOCK_SIZE_N']),
        B
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, out,
        B, C, H, W,
        out_channels, kH, kW,
        stride, padding,
        out_H, out_W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=1
    )
    
    # Add bias if provided
    if bias is not None:
        # Reshape bias for broadcasting: [1, out_channels, 1, 1]
        bias_view = bias.view(1, out_channels, 1, 1)
        out = out + bias_view
    
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Define the same convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        # Replace PyTorch's Conv2d with our Triton implementation
        # Note: For simplicity, we'll use the Triton kernel directly
        # but we need to extract weight and bias from the original conv1 layer
        
        # Use the weights and bias from the original layer
        weight = self.conv1.weight
        bias = self.conv1.bias
        
        # Call our custom Triton convolution
        return triton_conv2d(x, weight, bias, stride=4, padding=2)