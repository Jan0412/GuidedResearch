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
    # Get the batch and spatial indices
    batch_idx = tl.program_id(0)
    height_idx = tl.program_id(1)
    width_idx = tl.program_id(2)
    
    # Get the channel indices for this thread block
    in_channel_start = tl.program_id(3) * IN_CHANNELS_BLOCK
    out_channel_start = tl.program_id(4) * OUT_CHANNELS_BLOCK
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(IN_CHANNELS_BLOCK, 1))
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(OUT_CHANNELS_BLOCK, IN_CHANNELS_BLOCK))
    
    # Load weights for this output channel block
    for i in range(OUT_CHANNELS_BLOCK):
        if out_channel_start + i < out_channels:
            for j in range(IN_CHANNELS_BLOCK):
                if in_channel_start + j < in_channels:
                    shared_weight[i, j] = tl.load(weight_ptr + 
                        (out_channel_start + i) * in_channels + (in_channel_start + j))
    
    # Process input channels in chunks
    for in_ch_chunk in range((in_channels + IN_CHANNELS_BLOCK - 1) // IN_CHANNELS_BLOCK):
        in_ch_start = in_ch_chunk * IN_CHANNELS_BLOCK
        
        # Load input chunk into shared memory
        for i in range(IN_CHANNELS_BLOCK):
            if in_ch_start + i < in_channels:
                shared_input[i, 0] = tl.load(input_ptr + 
                    batch_idx * (in_channels * height * width) +
                    (in_ch_start + i) * (height * width) +
                    height_idx * width +
                    width_idx)
        
        # Compute dot product for each output channel
        for out_ch in range(OUT_CHANNELS_BLOCK):
            if out_channel_start + out_ch < out_channels:
                acc = tl.zeros((1,), dtype=tl.float32)
                for i in range(IN_CHANNELS_BLOCK):
                    if in_ch_start + i < in_channels:
                        acc += shared_input[i, 0] * shared_weight[out_ch, i]
                
                # Store result
                tl.store(output_ptr + 
                    batch_idx * (out_channels * height * width) +
                    (out_channel_start + out_ch) * (height * width) +
                    height_idx * width +
                    width_idx, 
                    acc[0])

def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    """
    Triton implementation of pointwise 2D convolution using shared memory optimization.
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda, "Weight tensor must be on CUDA"
    
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, height, width, device='cuda', dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 32
    IN_CHANNELS_BLOCK = 32
    OUT_CHANNELS_BLOCK = 32
    
    # Grid dimensions
    grid_batch = batch_size
    grid_height = height
    grid_width = width
    grid_in_channels = (in_channels + IN_CHANNELS_BLOCK - 1) // IN_CHANNELS_BLOCK
    grid_out_channels = (out_channels + OUT_CHANNELS_BLOCK - 1) // OUT_CHANNELS_BLOCK
    
    grid = (
        grid_batch,
        grid_height,
        grid_width,
        grid_in_channels,
        grid_out_channels
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
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Optimized version of the Model using Triton kernels for pointwise 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
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
        # Use our Triton implementation
        return triton_pointwise_conv2d(x, self.weight, self.bias_param)