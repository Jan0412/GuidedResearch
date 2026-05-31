import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    HEIGHT_PER_BLOCK: tl.constexpr,
    WIDTH_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    height_idx = tl.program_id(2)
    width_idx = tl.program_id(3)
    
    # Calculate global positions
    out_h_start = height_idx * HEIGHT_PER_BLOCK
    out_w_start = width_idx * WIDTH_PER_BLOCK
    
    # Shared memory for input tile and kernel
    input_tile = tl.shared_ptr(input_ptr, 
                              shape=[HEIGHT_PER_BLOCK + 2*padding, WIDTH_PER_BLOCK + 2*padding],
                              dtype=tl.float32)
    kernel_tile = tl.shared_ptr(weight_ptr, 
                               shape=[kernel_size, kernel_size],
                               dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Load kernel weights (assuming they're stored as [1, 1, kernel_size, kernel_size] for depthwise)
    kernel_offset = channel_idx * kernel_size * kernel_size
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            kernel_tile[k_h, k_w] = tl.load(weight_ptr + kernel_offset + k_h * kernel_size + k_w)
    
    # Process input tiles
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input position
            h_start = out_h_start * stride - padding + kh
            w_start = out_w_start * stride - padding + kw
            
            # Load input tile
            for h in range(HEIGHT_PER_BLOCK):
                for w in range(WIDTH_PER_BLOCK):
                    ih = h_start + h
                    iw = w_start + w
                    
                    if ih >= 0 and ih < height_in and iw >= 0 and iw < width_in:
                        input_tile[h, w] = tl.load(input_ptr + 
                                                  batch_idx * (in_channels * height_in * width_in) +
                                                  channel_idx * (height_in * width_in) +
                                                  ih * width_in + iw)
                    else:
                        input_tile[h, w] = 0.0
            
            # Compute convolution for this kernel position
            for h in range(HEIGHT_PER_BLOCK):
                for w in range(WIDTH_PER_BLOCK):
                    acc[h, w] += input_tile[h, w] * kernel_tile[kh, kw]
    
    # Store output
    for h in range(HEIGHT_PER_BLOCK):
        for w in range(WIDTH_PER_BLOCK):
            if out_h_start + h < height_out and out_w_start + w < width_out:
                out_offset = batch_idx * (in_channels * height_out * width_out) + \
                           channel_idx * (height_out * width_out) + \
                           (out_h_start + h) * width_out + (out_w_start + w)
                tl.store(output_ptr + out_offset, acc[h, w])

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution.
    """
    assert input_tensor.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert input_tensor.dtype == torch.float32 and weight.dtype == torch.float32, "Only FP32 supported"
    
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    _, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in + 2 * padding - kernel_size) // stride + 1
    width_out = (width_in + 2 * padding - kernel_size) // stride + 1
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 1024
    CHANNELS_PER_BLOCK = 1
    HEIGHT_PER_BLOCK = 8
    WIDTH_PER_BLOCK = 8
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        in_channels,
        (height_out + HEIGHT_PER_BLOCK - 1) // HEIGHT_PER_BLOCK,
        (width_out + WIDTH_PER_BLOCK - 1) // WIDTH_PER_BLOCK
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK=HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK=WIDTH_PER_BLOCK
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        """
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)