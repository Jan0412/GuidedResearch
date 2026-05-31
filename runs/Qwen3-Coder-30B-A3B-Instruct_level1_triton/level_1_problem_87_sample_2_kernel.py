import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def pointwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_BIAS: tl.constexpr
):
    # Calculate global thread index
    pid = tl.program_id(0)
    
    # Each block processes one output element
    # We'll use a 2D grid where each thread processes one output element
    # But we can also fuse multiple operations
    
    # For simplicity, let's process one output element per thread
    # In practice, you'd want to process multiple elements or use a more complex tiling strategy
    output_idx = pid
    
    # Convert linear index to 4D coordinates (batch, out_channel, h, w)
    out_w = output_idx % width
    out_h = (output_idx // width) % height
    out_c = (output_idx // (width * height)) % out_channels
    batch_idx = (output_idx // (width * height * out_channels)) % batch_size
    
    if output_idx >= batch_size * out_channels * height * width:
        return
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(in_channels):
        # Calculate input position
        input_h = out_h * stride_h - padding_h
        input_w = out_w * stride_w - padding_w
        
        # Get input value (assuming kernel size is 1x1, so no sliding window)
        # For a 1x1 conv, we just do a matrix multiplication-like operation
        input_val = tl.load(input_ptr + 
                           batch_idx * (in_channels * height * width) +
                           ic * (height * width) +
                           input_h * width + input_w)
        
        # Get weight value (for 1x1 conv, it's a simple lookup)
        weight_val = tl.load(weight_ptr + 
                            out_c * in_channels + ic)
        
        acc += input_val * weight_val
    
    # Add bias if needed
    if USE_BIAS:
        bias_val = tl.load(bias_ptr + out_c)
        acc += bias_val
    
    # Store output
    tl.store(output_ptr + 
             batch_idx * (out_channels * height * width) +
             out_c * (height * width) +
             out_h * width + out_w,
             acc)

def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    """
    Triton implementation of pointwise 2D convolution (1x1 conv)
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda, "Weight tensor must be on CUDA"
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, height, width, device=input_tensor.device, dtype=torch.float32)
    
    # Handle bias
    use_bias = bias is not None
    if use_bias:
        bias = bias.contiguous()
    
    # Calculate grid size
    total_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = lambda meta: (math.ceil(total_elements / meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        stride_h=1,
        stride_w=1,
        padding_h=0,
        padding_w=0,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_BIAS=use_bias
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bias = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the Triton kernel for the convolution operation
        weight = self.conv1d.weight.data
        bias = self.conv1d.bias.data if self.bias else None
        
        # Convert to float32 for Triton kernel compatibility
        x = x.to(torch.float32)
        weight = weight.to(torch.float32)
        if bias is not None:
            bias = bias.to(torch.float32)
            
        return triton_pointwise_conv2d(x, weight, bias)