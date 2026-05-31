import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # Get program ID and create block index
    pid = tl.program_id(0)
    group_id = pid // GROUP_SIZE
    group_size = min(GROUP_SIZE, output_length - group_id * GROUP_SIZE)
    
    # Each program handles one output position
    if group_id * GROUP_SIZE + tl.program_id(1) >= output_length:
        return
        
    output_pos = group_id * GROUP_SIZE + tl.program_id(1)
    
    # Calculate output channel
    out_channel = tl.program_id(2)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for ic in range(in_channels):
        for k in range(kernel_size):
            # Calculate input position
            input_pos = (output_pos - padding + k * dilation) // stride
            
            # Check bounds
            if (input_pos >= 0 and input_pos < input_length and 
                (output_pos - padding + k * dilation) % stride == 0):
                
                # Load input value
                input_val = tl.load(input_ptr + 
                                  (0 * in_channels * input_length + 
                                   ic * input_length + 
                                   input_pos))
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                   (out_channel * in_channels * kernel_size + 
                                    ic * kernel_size + 
                                    k))
                
                acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + 
             (0 * out_channels * output_length + 
              out_channel * output_length + 
              output_pos), acc)

def triton_conv1d_transpose(input_tensor, weight, bias, stride, padding, dilation):
    """
    Triton implementation of ConvTranspose1d
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Handle bias
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Define grid dimensions
    grid = (
        triton.cdiv(output_length, 128),  # Number of blocks for output positions
        1,  # Only one block per output position
        out_channels  # One block per output channel
    )
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
        input_tensor,
        weight,
        output,
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
        BLOCK_SIZE=128,
        GROUP_SIZE=128
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.
    Optimized with custom Triton kernels.
    """
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
        Performs the transposed 1D convolution using Triton kernel.
        """
        return triton_conv1d_transpose(x, self.weight, self.bias, self.stride, self.padding, self.dilation)