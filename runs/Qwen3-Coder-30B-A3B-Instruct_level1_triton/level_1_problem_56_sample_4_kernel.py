import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    out_channel_id = tl.program_id(2)
    
    # Calculate global indices
    batch_offset = batch_id * in_channels * input_height * input_width
    weight_offset = out_channel_id * in_channels // groups * kernel_height * kernel_width
    bias_offset = out_channel_id
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        if g != group_id:
            continue
            
        # Calculate input and weight offsets for this group
        group_input_offset = batch_offset + g * (in_channels // groups) * input_height * input_width
        group_weight_offset = weight_offset + g * (in_channels // groups) * kernel_height * kernel_width
        
        # Loop over kernel elements
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position
                ih = kh * dilation_h - padding_h
                iw = kw * dilation_w - padding_w
                
                # Loop over output spatial positions
                for oh in range(output_height):
                    for ow in range(output_width):
                        # Calculate input coordinates
                        ih_pos = oh * stride_h + ih
                        iw_pos = ow * stride_w + iw
                        
                        # Check bounds
                        if ih_pos >= 0 and ih_pos < input_height and iw_pos >= 0 and iw_pos < input_width:
                            # Calculate input index
                            input_idx = group_input_offset + (ih_pos * input_width + iw_pos) * (in_channels // groups)
                            
                            # Load input value
                            input_val = tl.load(input_ptr + input_idx)
                            
                            # Load weight value
                            weight_idx = group_weight_offset + kh * kernel_width + kw
                            weight_val = tl.load(weight_ptr + weight_idx)
                            
                            # Accumulate
                            acc += input_val * weight_val
    
    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + bias_offset)
        acc += bias_val
    
    # Store result
    if out_channel_id < out_channels and batch_id < batch_size:
        output_idx = batch_id * out_channels * output_height * output_width + out_channel_id * output_height * output_width
        output_idx += oh * output_width + ow
        tl.store(output_ptr + output_idx, acc)

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of 2D convolution
    """
    # Get tensor shapes
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Prepare grid configuration
    grid = (
        batch_size,           # batch dimension
        groups,               # group dimension
        out_channels          # output channel dimension
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        BLOCK_SIZE=1024,
        GROUPS_PER_BLOCK=1,
        CHANNELS_PER_BLOCK=1
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Convert to float32 if needed
        if x.dtype != torch.float32:
            x = x.float()
            
        # Call the Triton-based convolution
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )

# Note: The above implementation has limitations compared to PyTorch's full convolution.
# A more optimized version would require implementing proper tiling and shared memory usage
# for better performance on GPUs. For production use, PyTorch's native conv2d is recommended.