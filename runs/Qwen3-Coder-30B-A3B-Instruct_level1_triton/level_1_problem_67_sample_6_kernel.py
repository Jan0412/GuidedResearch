import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv1d_kernel(
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
    groups,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    out_pos_id = tl.program_id(2)
    
    # Calculate output position
    out_pos = out_pos_id * BLOCK_N
    
    # Shared memory for input window and weights
    input_window = tl.shared_pointer(input_ptr + batch_id * in_channels * input_length + out_pos * stride - padding, 
                                    shape=(in_channels, kernel_size + 2 * padding), dtype=tl.float32)
    weights = tl.shared_pointer(weight_ptr + out_channel_id * in_channels * kernel_size + out_pos_id * BLOCK_N, 
                               shape=(in_channels, kernel_size), dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over groups and compute convolution
    for g in range(groups):
        group_offset = g * (in_channels // groups)
        group_weight_offset = out_channel_id * (in_channels // groups) * kernel_size
        
        # Load weights for this group
        for k in range(0, kernel_size, BLOCK_K):
            # Load weights chunk
            weight_chunk = tl.load(weight_ptr + group_weight_offset + k, 
                                 mask=(k + tl.arange(0, BLOCK_K)) < kernel_size, other=0.0)
            
            # Load input window chunk
            input_chunk = tl.load(input_ptr + batch_id * in_channels * input_length + 
                                (out_pos * stride - padding + k), 
                                mask=(out_pos * stride - padding + k) < input_length, other=0.0)
            
            # Compute partial dot product
            acc += tl.dot(input_chunk, weight_chunk)
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_id)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + batch_id * out_channels * output_length + 
             out_channel_id * output_length + out_pos, acc)

def triton_conv1d(input_tensor, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton-based 1D convolution implementation
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Define block sizes
    BLOCK_M = 1
    BLOCK_N = 32
    BLOCK_K = 32
    
    # Grid dimensions
    grid = (
        batch_size,
        out_channels,
        (output_length + BLOCK_N - 1) // BLOCK_N
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        """
        return triton_conv1d(x, self.weight, self.bias, 
                           stride=self.stride, 
                           padding=self.padding, 
                           dilation=self.dilation, 
                           groups=self.groups)