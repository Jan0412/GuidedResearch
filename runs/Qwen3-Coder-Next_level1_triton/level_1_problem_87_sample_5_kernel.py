import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pointwise_conv_kernel(
    x_ptr,  # Input tensor pointer (B, C_in, H, W)
    w_ptr,  # Weight tensor pointer (C_out, C_in)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (B, C_out, H, W)
    B, C_in, C_out, H, W,
    stride_x_batch, stride_x_channel, stride_x_height, stride_x_width,
    stride_w_out, stride_w_in,
    stride_out_batch, stride_out_channel, stride_out_height, stride_out_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes a subset of output channels
    out_channel_block = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    # Calculate batch and spatial coordinates from spatial_idx
    # spatial_idx ranges from 0 to B*H*W-1
    batch_idx = spatial_idx // (H * W)
    hw_idx = spatial_idx % (H * W)
    h = hw_idx // W
    w = hw_idx % W
    
    # Compute starting offsets for this thread's data
    x_batch_offset = batch_idx * stride_x_batch
    x_h_offset = h * stride_x_height
    x_w_offset = w * stride_x_width
    
    out_batch_offset = batch_idx * stride_out_batch
    out_h_offset = h * stride_out_height
    out_w_offset = w * stride_out_width
    
    # Accumulate over input channels
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process input channels in blocks
    for start_c_in in range(0, C_in, BLOCK_SIZE):
        c_in_offsets = start_c_in + tl.arange(0, BLOCK_SIZE)
        c_in_mask = c_in_offsets < C_in
        
        # Load input values: shape (BLOCK_SIZE,)
        x_offsets = (x_batch_offset + 
                    c_in_offsets * stride_x_channel + 
                    x_h_offset + x_w_offset)
        x_vals = tl.load(x_ptr + x_offsets, mask=c_in_mask, other=0.0)
        
        # Load weight values: shape (BLOCK_SIZE,)
        w_offsets = (out_channel_block * stride_w_out + 
                    c_in_offsets * stride_w_in)
        w_vals = tl.load(w_ptr + w_offsets, mask=c_in_mask, other=0.0)
        
        # Accumulate
        acc += x_vals * w_vals
    
    # Convert accumulator to float16 if needed
    acc = acc.to(tl.float16)
    
    # Store result
    out_offsets = (out_batch_offset + 
                  out_channel_block * stride_out_channel + 
                  out_h_offset + out_w_offset)
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_channel_block)
        acc = acc + bias
    
    tl.store(out_ptr + out_offsets, acc)


def triton_pointwise_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    Performs pointwise (1x1) convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (B, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in)
        bias: Optional bias tensor of shape (C_out,)
        
    Returns:
        Output tensor of shape (B, C_out, H, W)
    """
    B, C_in, H, W = x.shape
    C_out, _ = weight.shape
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    out = torch.empty(B, C_out, H, W, dtype=x.dtype, device=x.device)
    
    # Calculate strides
    stride_x = x.stride()
    stride_w = weight.stride()
    stride_out = out.stride()
    
    # Grid configuration: one block per output channel, one block per spatial location
    # We'll process spatial locations in batches for better performance
    BLOCK_SIZE = 32  # Tune this for performance
    grid = (C_out, B * H * W)
    
    # Launch kernel
    pointwise_conv_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, H, W,
        *stride_x, *stride_w, *stride_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for pointwise convolution.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use the same convolution layer but replace forward pass with Triton kernel
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        # Store the original parameters for potential use
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Extract weight and bias from the convolution layer
        weight = self.conv1d.weight
        bias = self.conv1d.bias
        
        # Use the Triton kernel for pointwise convolution
        return triton_pointwise_conv(x, weight, bias)