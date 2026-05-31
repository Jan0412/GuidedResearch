import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_size,
    stride,
    padding,
    groups,
    output_padding,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    HEIGHT_PER_BLOCK: tl.constexpr,
    WIDTH_PER_BLOCK: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_channel_idx = tl.program_id(2)
    
    # Calculate channel offset for this group
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    
    # Shared memory for input tile
    input_tile = tl.shared_tensor(tl.float32, (HEIGHT_PER_BLOCK + 2*padding, WIDTH_PER_BLOCK + 2*padding))
    
    # Initialize accumulator
    acc = tl.zeros((HEIGHT_PER_BLOCK, WIDTH_PER_BLOCK), dtype=tl.float32)
    
    # Loop over kernel
    for k in range(channels_per_group):
        # Load input tile with padding
        for i in range(HEIGHT_PER_BLOCK + 2*padding):
            for j in range(WIDTH_PER_BLOCK + 2*padding):
                h_in = i - padding
                w_in = j - padding
                
                if h_in >= 0 and h_in < height_in and w_in >= 0 and w_in < width_in:
                    input_val = tl.load(input_ptr + 
                                       batch_idx * (in_channels * height_in * width_in) +
                                       (group_idx * channels_per_group + k) * (height_in * width_in) +
                                       h_in * width_in + w_in)
                else:
                    input_val = 0.0
                    
                input_tile[i, j] = input_val
        
        # Apply kernel
        for i in range(HEIGHT_PER_BLOCK):
            for j in range(WIDTH_PER_BLOCK):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        h_out = i * stride + kh - padding
                        w_out = j * stride + kw - padding
                        
                        if h_out >= 0 and h_out < height_out and w_out >= 0 and w_out < width_out:
                            weight_val = tl.load(weight_ptr + 
                                               (out_channel_idx * groups + group_idx) * (channels_per_group * kernel_size * kernel_size) +
                                               k * (kernel_size * kernel_size) +
                                               kh * kernel_size + kw)
                            
                            input_val = input_tile[i + kh, j + kw]
                            acc[i, j] += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val
    
    # Store output
    for i in range(HEIGHT_PER_BLOCK):
        for j in range(WIDTH_PER_BLOCK):
            h_out = i * stride
            w_out = j * stride
            
            if h_out < height_out and w_out < width_out:
                tl.store(output_ptr + 
                        batch_idx * (out_channels * height_out * width_out) +
                        out_channel_idx * (height_out * width_out) +
                        h_out * width_out + w_out,
                        acc[i, j])

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding, output_padding, groups):
    """
    Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding[0]
    width_out = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding[1]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Define block sizes
    BLOCK_SIZE = 256
    GROUP_SIZE = 8
    CHANNELS_PER_BLOCK = 1
    HEIGHT_PER_BLOCK = 16
    WIDTH_PER_BLOCK = 16
    
    # Grid configuration
    grid = (
        batch_size,
        groups,
        out_channels
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_size,
        stride,
        padding,
        groups,
        output_padding,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        HEIGHT_PER_BLOCK=HEIGHT_PER_BLOCK,
        WIDTH_PER_BLOCK=WIDTH_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.groups = groups
        self.bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.groups
        )