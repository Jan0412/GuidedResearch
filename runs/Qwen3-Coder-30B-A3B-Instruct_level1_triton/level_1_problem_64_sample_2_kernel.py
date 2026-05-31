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
    input_size,
    output_size,
    weight_size,
    batch_size,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
    OUT_CHANNELS: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_pos_idx = tl.program_id(2)
    
    # Shared memory for weight loading
    shared_weight = tl.shared_ptr(weight_ptr, shape=(GROUPS, OUT_CHANNELS // GROUPS, KERNEL_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Load weight for this group and channel
    if out_pos_idx < OUTPUT_SIZE:
        for k in range(KERNEL_SIZE):
            # Calculate input position
            input_pos = out_pos_idx - padding + k * stride
            
            # Check bounds
            if input_pos >= 0 and input_pos < input_size:
                # Load input value
                input_val = tl.load(input_ptr + 
                                   batch_idx * (in_channels * input_size) +
                                   out_ch_idx * input_size +
                                   input_pos)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                    out_ch_idx * kernel_size + 
                                    k)
                
                acc += input_val * weight_val
    
    # Store result
    if out_pos_idx < OUTPUT_SIZE:
        tl.store(output_ptr + 
                batch_idx * (out_channels * output_size) +
                out_ch_idx * output_size +
                out_pos_idx, 
                acc)

def triton_conv1d_transpose(input_tensor, weight, bias=None, stride=1, padding=0, groups=1):
    """
    Triton implementation of ConvTranspose1d
    """
    batch_size, in_channels, input_size = input_tensor.shape
    out_channels, kernel_size = weight.shape
    
    # Calculate output size
    output_size = (input_size - 1) * stride - 2 * padding + kernel_size
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_size, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_size
    )
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
        input_tensor,
        weight,
        output,
        input_size,
        output_size,
        kernel_size,
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS=groups,
        OUTPUT_SIZE=output_size,
        KERNEL_SIZE=kernel_size,
        IN_CHANNELS=in_channels,
        OUT_CHANNELS=out_channels
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1)
    
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights with Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        """
        # Use Triton implementation instead of PyTorch's native implementation
        return triton_conv1d_transpose(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            groups=self.groups
        )

# Keep the original test functions for compatibility
def get_inputs():
    batch_size = 64
    in_channels = 128
    out_channels = 128
    kernel_size = 3
    length = 65536
    x = torch.rand(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [128, 128, 3]  # in_channels, out_channels, kernel_size