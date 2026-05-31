import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    groups,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate global output element index
    output_elem_idx = pid * OUTPUT_ELEMENTS_PER_BLOCK
    
    # Calculate which output element this thread block is working on
    if output_elem_idx >= batch_size * out_channels * output_depth * output_height * output_width:
        return
    
    # Decompose the output element index into batch, channel, depth, height, width
    elem_remaining = output_elem_idx
    out_w = elem_remaining % output_width
    elem_remaining //= output_width
    out_h = elem_remaining % output_height
    elem_remaining //= output_height
    out_d = elem_remaining % output_depth
    elem_remaining //= output_depth
    out_c = elem_remaining % out_channels
    elem_remaining //= out_channels
    batch_idx = elem_remaining
    
    # Calculate group information
    group_idx = out_c // CHANNELS_PER_GROUP
    
    # Calculate input coordinates for this output position
    # For transposed conv, we need to reverse the logic
    # Input coordinate corresponding to output position (out_d, out_h, out_w)
    input_d_start = out_d - kernel_size + 1 + padding
    input_h_start = out_h - kernel_size + 1 + padding
    input_w_start = out_w - kernel_size + 1 + padding
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over kernel
    for k_d in range(kernel_size):
        for k_h in range(kernel_size):
            for k_w in range(kernel_size):
                # Calculate input coordinates
                input_d = input_d_start + k_d
                input_h = input_h_start + k_h
                input_w = input_w_start + k_w
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and 
                    input_h >= 0 and input_h < input_height and 
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input channel offset within group
                    input_c_offset = (out_c % CHANNELS_PER_GROUP) * groups
                    
                    # Calculate input and weight indices
                    input_idx = (batch_idx * in_channels * input_depth * input_height * input_width +
                                input_c_offset * input_depth * input_height * input_width +
                                input_d * input_height * input_width +
                                input_h * input_width +
                                input_w)
                    
                    weight_idx = (out_c * in_channels * kernel_size * kernel_size * kernel_size +
                                 input_c_offset * kernel_size * kernel_size * kernel_size +
                                 k_d * kernel_size * kernel_size +
                                 k_h * kernel_size +
                                 k_w)
                    
                    # Load values and accumulate
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    acc += input_val * weight_val
    
    # Store result
    output_idx = (batch_idx * out_channels * output_depth * output_height * output_width +
                 out_c * output_depth * output_height * output_width +
                 out_d * output_height * output_width +
                 out_h * output_width +
                 out_w)
    
    tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=1, padding=0, groups=1):
    """
    Triton implementation of 3D transposed convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride - 2 * padding + kernel_depth
    output_height = (input_height - 1) * stride - 2 * padding + kernel_height
    output_width = (input_width - 1) * stride - 2 * padding + kernel_width
    
    # Allocate output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous and on GPU
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Set up kernel parameters
    CHANNELS_PER_GROUP = out_channels // groups
    OUTPUT_ELEMENTS_PER_BLOCK = 256
    BLOCK_SIZE = 256
    
    # Calculate grid size
    total_output_elements = batch_size * out_channels * output_depth * output_height * output_width
    grid_size = (total_output_elements + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
    
    # Launch kernel
    conv_transpose3d_kernel[grid_size](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_depth,
        stride,
        padding,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_GROUP=CHANNELS_PER_GROUP,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            groups=self.groups
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, output_padding={self.output_padding}, "
            f"groups={self.groups}, bias={self.bias is not None}"
        )