import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H, W)
    w_ptr,  # Weight tensor: (C_out, C_in, 1, 1)
    b_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (B, C_out, H, W)
    B, C_in, C_out, H, W,
    stride_x_batch, stride_x_channel, stride_x_height, stride_x_width,
    stride_w_out, stride_w_in, stride_w_kh, stride_w_kw,
    stride_out_batch, stride_out_channel, stride_out_height, stride_out_width,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for spatial elements
):
    # Compute batch index
    batch_idx = tl.program_id(0)
    # Compute output channel block index
    out_c_block_idx = tl.program_id(1)
    
    # Compute the start index for output channels
    out_c_start = out_c_block_idx * BLOCK_SIZE_M
    out_c_offsets = out_c_start + tl.arange(0, BLOCK_SIZE_M)
    out_c_mask = out_c_offsets < C_out
    
    # For each spatial position (H*W), process in blocks
    for h_idx in range(0, H):
        for w_idx in range(0, W):
            # Compute input and output indices
            spatial_idx = h_idx * W + w_idx
            in_c_start = 0
            
            # Accumulate over input channels
            acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            
            # Loop over input channels in blocks
            for in_c_block in range(0, (C_in + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N):
                in_c_block_start = in_c_block * BLOCK_SIZE_N
                in_c_offsets = in_c_block_start + tl.arange(0, BLOCK_SIZE_N)
                in_c_mask = in_c_offsets < C_in
                
                # Load input: x[batch_idx, in_c_offsets, h_idx, w_idx]
                x_offsets = (batch_idx * stride_x_batch + 
                            in_c_offsets * stride_x_channel + 
                            h_idx * stride_x_height + 
                            w_idx * stride_x_width)
                x_val = tl.load(x_ptr + x_offsets, mask=in_c_mask, other=0.0)
                
                # Load weights: w[out_c_offsets, in_c_offsets, 0, 0]
                w_offsets = (out_c_offsets[:, None] * stride_w_out + 
                            in_c_offsets[None, :] * stride_w_in + 
                            0 * stride_w_kh + 
                            0 * stride_w_kw)
                w_val = tl.load(w_ptr + w_offsets, mask=out_c_mask[:, None] and in_c_mask[None, :], other=0.0)
                
                # Accumulate: acc[out_c] += sum over in_c of x[in_c] * w[out_c, in_c]
                # x_val is (BLOCK_SIZE_N,), w_val is (BLOCK_SIZE_M, BLOCK_SIZE_N)
                # So we need: acc += x_val @ w_val.T
                acc += tl.sum(x_val[None, :] * w_val, axis=1)
            
            # Add bias if provided
            if b_ptr is not None:
                b_offsets = out_c_offsets
                b_val = tl.load(b_ptr + b_offsets, mask=out_c_mask, other=0.0)
                acc += b_val
            
            # Store result
            out_offsets = (batch_idx * stride_out_batch + 
                          out_c_offsets * stride_out_channel + 
                          h_idx * stride_out_height + 
                          w_idx * stride_out_width)
            tl.store(out_ptr + out_offsets, acc.to(x_ptr.dtype.element_ty), mask=out_c_mask)

def triton_pointwise_conv2d(x, weight, bias=None):
    """
    Performs pointwise 2D convolution (1x1 conv) using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, 1, 1)
        bias: Optional bias tensor of shape (out_channels,)
    
    Returns:
        Output tensor of shape (batch_size, out_channels, height, width)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    B, C_in, H, W = x.shape
    C_out, _, _, _ = weight.shape
    
    # Prepare output tensor
    out = torch.empty(B, C_out, H, W, device=x.device, dtype=x.dtype)
    
    # Compute strides
    stride_x = x.stride()
    stride_w = weight.stride()
    stride_out = out.stride()
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_M = 32  # Output channels per block
    BLOCK_SIZE_N = 32  # Input channels per block
    BLOCK_SIZE_K = 1   # Spatial elements per block (processed sequentially)
    
    # Grid: (batch_size, number of output channel blocks)
    grid = lambda meta: (B, (C_out + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'])
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, H, W,
        stride_x[0], stride_x[1], stride_x[2], stride_x[3],
        stride_w[0], stride_w[1], stride_w[2], stride_w[3],
        stride_out[0], stride_out[1], stride_out[2], stride_out[3],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized pointwise 2D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using optimized Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Use our optimized Triton implementation for pointwise convolution
        return triton_pointwise_conv2d(x, self.conv1d.weight, self.conv1d.bias)