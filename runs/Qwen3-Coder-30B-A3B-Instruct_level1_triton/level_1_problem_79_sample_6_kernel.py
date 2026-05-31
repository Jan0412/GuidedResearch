import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv1d_transpose_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Get thread and block indices
    pid = tl.program_id(0)
    batch_pid = tl.program_id(1)
    
    # Calculate which output elements this block handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate output position for this block
    output_pos = block_start
    
    # Shared memory for input tiles
    input_tile = tl.shared_memory(dtype=tl.float32, shape=(GROUP_SIZE, BLOCK_SIZE))
    
    # Loop over kernel elements
    for k in range(0, in_channels, GROUP_SIZE):
        # Load input tile
        input_offsets = k + tl.arange(0, GROUP_SIZE)[:, None] * input_length + tl.arange(0, BLOCK_SIZE)[None, :]
        input_mask = (k + tl.arange(0, GROUP_SIZE)[:, None] < in_channels) & (tl.arange(0, BLOCK_SIZE)[None, :] < input_length)
        input_data = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        
        # Load weights for this channel group
        weight_offsets = k + tl.arange(0, GROUP_SIZE)[:, None] * out_channels + tl.arange(0, BLOCK_SIZE)[None, :] * kernel_size
        weight_mask = (k + tl.arange(0, GROUP_SIZE)[:, None] < in_channels) & (tl.arange(0, BLOCK_SIZE)[None, :] < out_channels * kernel_size)
        weight_data = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
        
        # Perform convolution
        for i in range(kernel_size):
            # Calculate output positions for this kernel element
            pos = output_pos + i * dilation
            # Apply stride and padding
            start_pos = pos - padding
            if start_pos >= 0 and start_pos < output_length:
                # Compute contribution to output
                for j in range(GROUP_SIZE):
                    if k + j < in_channels:
                        output_idx = batch_pid * out_channels * output_length + (k + j) * output_length + pos
                        # Only process valid outputs
                        if pos < output_length:
                            tl.atomic_add(output_ptr + output_idx, input_data[j, 0] * weight_data[j, 0])
    
    # Add bias if available
    if bias_ptr is not None:
        bias_offsets = tl.arange(0, BLOCK_SIZE) + batch_pid * out_channels
        bias_mask = tl.arange(0, BLOCK_SIZE) < out_channels
        bias_data = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
        for i in range(BLOCK_SIZE):
            if i < out_channels:
                output_offset = batch_pid * out_channels * output_length + i * output_length + output_pos
                tl.atomic_add(output_ptr + output_offset, bias_data[i])

def triton_conv1d_transpose(input_tensor, weight, bias, stride, padding, dilation):
    """
    Custom Triton implementation of ConvTranspose1d
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    
    # Initialize output tensor
    output = torch.zeros(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Flatten tensors for easier handling
    input_flat = input_tensor.view(-1, input_length)
    weight_flat = weight.view(out_channels, -1)
    output_flat = output.view(-1, output_length)
    
    # Grid configuration
    grid_size = (math.ceil(output_length / 128), batch_size)
    
    # Launch kernel
    conv1d_transpose_kernel[grid_size](
        input_flat,
        weight_flat,
        output_flat,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation,
        BLOCK_SIZE=128,
        GROUP_SIZE=32
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using custom Triton kernel.
        """
        return triton_conv1d_transpose(x, self.weight, self.bias, self.stride, self.padding, self.dilation)