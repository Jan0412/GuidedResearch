import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, length)
    w_ptr,  # Weight tensor: (in_channels, out_channels // groups, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, out_length)
    batch_size, in_channels, out_channels, kernel_size,
    stride, padding, output_padding, groups,
    in_length, out_length,
    BLOCK_SIZE: tl.constexpr
):
    # Compute output position
    out_idx = tl.program_id(0)
    batch_id = out_idx // (out_channels * out_length)
    out_channel = (out_idx // out_length) % out_channels
    out_pos = out_idx % out_length
    
    # Compute input channel group
    group_size = in_channels // groups
    group_id = out_channel // (out_channels // groups)
    in_channel_start = group_id * group_size
    
    # Accumulator for the result
    acc = 0.0
    
    # Iterate over kernel positions
    for k in range(kernel_size):
        # Compute input position considering stride and padding
        in_pos = out_pos * stride + k - padding
        
        # Check if within valid input range
        if in_pos >= 0 and in_pos < in_length:
            # Compute input channel index
            for c in range(group_size):
                in_c = in_channel_start + c
                # Compute indices for 1D tensor access
                x_offset = batch_id * (in_channels * in_length) + in_c * in_length + in_pos
                w_offset = in_c * (out_channels // groups * kernel_size) + (out_channel % (out_channels // groups)) * kernel_size + k
                
                # Load values and accumulate
                x_val = tl.load(x_ptr + x_offset)
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + out_channel)
        acc += b_val
    
    # Store result
    tl.store(out_ptr + out_idx, acc)


def triton_conv_transpose1d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose1d
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, in_length = x.shape
    out_channels = weight.shape[1] * groups  # weight shape: (in_channels, out_channels // groups, kernel_size)
    kernel_size = weight.shape[2]
    
    # Calculate output length
    out_length = (in_length - 1) * stride - 2 * padding + output_padding + kernel_size
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_length, device=x.device, dtype=x.dtype)
    
    # Flatten the output tensor for 1D indexing
    n_elements = batch_size * out_channels * out_length
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch grid
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, kernel_size,
        stride, padding, output_padding, groups,
        in_length, out_length,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation using custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Create weight and bias parameters (same as nn.ConvTranspose1d)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights using the same initialization as nn.ConvTranspose1d
        self.reset_parameters()
    
    def reset_parameters(self):
        # Same initialization as nn.ConvTranspose1d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call our Triton-based transposed convolution
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, 
            output_padding=self.output_padding, groups=self.groups
        )