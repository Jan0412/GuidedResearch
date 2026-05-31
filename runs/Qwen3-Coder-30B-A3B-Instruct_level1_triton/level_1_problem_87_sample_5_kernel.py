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
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
    IN_CHANNELS_BLOCK: tl.constexpr,
    OUT_CHANNELS_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)
    
    # Calculate global indices
    global_h = h_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    global_w = w_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid indices
    h_mask = global_h < height
    w_mask = global_w < width
    
    # Load input slice for this batch and channel
    input_offset = batch_idx * in_channels * height * width + \
                   tl.arange(0, in_channels)[:, None, None] * height * width + \
                   global_h[None, :, None] * width + \
                   global_w[None, None, :]
    
    # Load weights for this output channel
    weight_offset = out_ch_idx * in_channels + tl.arange(0, in_channels)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Perform dot product across input channels
    for c in range(0, in_channels, IN_CHANNELS_BLOCK):
        # Load input chunk
        input_chunk = tl.load(input_ptr + input_offset + c * height * width, 
                             mask=(tl.arange(0, IN_CHANNELS_BLOCK)[:, None, None] < in_channels - c) &
                                  h_mask[None, :, None] & w_mask[None, None, :],
                             other=0.0)
        
        # Load weight chunk
        weight_chunk = tl.load(weight_ptr + weight_offset + c, 
                              mask=tl.arange(0, IN_CHANNELS_BLOCK) < in_channels - c,
                              other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(input_chunk * weight_chunk[:, None, None], axis=0)
    
    # Write output
    output_offset = batch_idx * out_channels * height * width + \
                    out_ch_idx * height * width + \
                    global_h[:, None] * width + \
                    global_w[None, :]
    
    tl.store(output_ptr + output_offset, acc, 
             mask=h_mask[:, None] & w_mask[None, :])

def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    """
    Triton implementation of pointwise 2D convolution
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda, "Weight tensor must be on CUDA"
    
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, _, _ = weight.shape
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, height, width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 16
    IN_CHANNELS_BLOCK = 32
    OUT_CHANNELS_BLOCK = 16
    
    # Grid configuration
    grid = (
        batch_size,  # batch dimension
        out_channels,  # output channels
        (height + BLOCK_SIZE - 1) // BLOCK_SIZE,  # height tiles
        (width + BLOCK_SIZE - 1) // BLOCK_SIZE   # width tiles
    )
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE,
        IN_CHANNELS_BLOCK=IN_CHANNELS_BLOCK,
        OUT_CHANNELS_BLOCK=OUT_CHANNELS_BLOCK
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation using Triton optimization.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_param', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return triton_pointwise_conv2d(x, self.weight, self.bias_param)