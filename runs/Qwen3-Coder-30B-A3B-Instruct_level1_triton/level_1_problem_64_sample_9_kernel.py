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
    batch_size,
    in_channels,
    out_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    
    # Calculate group information
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    
    # Shared memory for weight caching
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(GROUP_SIZE, kernel_size))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate indices for current group
        group_in_start = g * channels_per_group
        group_out_start = g * out_channels_per_group
        
        # Check if this thread block handles this group
        if out_channel_id >= group_out_start and out_channel_id < group_out_start + out_channels_per_group:
            # Load weight for this channel and group
            weight_offset = group_in_start * out_channels_per_group + (out_channel_id - group_out_start) * kernel_size
            weight_base = weight_ptr + weight_offset
            
            # Load weights into shared memory
            for i in range(0, kernel_size, GROUP_SIZE):
                if i + tl.program_id(2) < kernel_size:
                    shared_weight[tl.program_id(2), i + tl.program_id(2)] = tl.load(weight_base + i + tl.program_id(2))
            
            tl.sync()
            
            # Process input chunks
            for k in range(0, input_length, BLOCK_SIZE):
                # Load input data
                input_offset = batch_id * in_channels * input_length + group_in_start * input_length + k
                input_base = input_ptr + input_offset
                
                # Calculate output positions
                output_start = k * stride - padding
                output_end = min(output_start + BLOCK_SIZE * stride, output_length)
                
                # Process each kernel position
                for i in range(kernel_size):
                    if i < kernel_size:
                        weight_val = shared_weight[tl.program_id(2), i]
                        # For transposed conv, we need to map from output back to input
                        for j in range(BLOCK_SIZE):
                            pos = k + j
                            if pos < input_length:
                                # Calculate corresponding output position
                                out_pos = pos * stride - padding + i
                                if out_pos >= 0 and out_pos < output_length:
                                    input_val = tl.load(input_base + pos, mask=(pos < input_length))
                                    acc[j] += input_val * weight_val
            
            # Write output
            output_offset = batch_id * out_channels * output_length + out_channel_id * output_length
            output_base = output_ptr + output_offset
            
            for i in range(BLOCK_SIZE):
                if k + i < output_length:
                    tl.store(output_base + k + i, acc[i], mask=(k + i < output_length))

# Optimized version with better memory access patterns
@triton.jit
def conv1d_transpose_fused_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    group_id = tl.program_id(2)
    
    # Calculate group sizes
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    
    # Calculate tile boundaries
    tile_m = tl.minimum(BLOCK_SIZE_M, output_length - batch_id * output_length)
    tile_n = tl.minimum(BLOCK_SIZE_N, out_channels_per_group)
    
    # Shared memory for tiles
    a_tile = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_tile = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Process groups
    if group_id < groups:
        group_in_start = group_id * channels_per_group
        group_out_start = group_id * out_channels_per_group
        
        # Load bias if exists
        bias_val = 0.0
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + out_channel_id)
        
        # Process input-output pairs
        for k in range(0, input_length, BLOCK_SIZE_K):
            # Load input tile
            input_offset = batch_id * in_channels * input_length + group_in_start * input_length + k
            input_base = input_ptr + input_offset
            
            # Load weight tile
            weight_offset = group_in_start * out_channels_per_group + (out_channel_id - group_out_start) * kernel_size
            weight_base = weight_ptr + weight_offset
            
            # Compute partial dot product
            for m in range(0, BLOCK_SIZE_M, BLOCK_SIZE_K):
                for n in range(0, BLOCK_SIZE_N, BLOCK_SIZE_K):
                    # Load input tile
                    a_load_mask = (m + tl.arange(0, BLOCK_SIZE_K)) < input_length
                    a_load_offset = batch_id * in_channels * input_length + group_in_start * input_length + (m + tl.arange(0, BLOCK_SIZE_K))
                    a_tile = tl.load(input_ptr + a_load_offset, mask=a_load_mask)
                    
                    # Load weight tile  
                    b_load_mask = (n + tl.arange(0, BLOCK_SIZE_K)) < kernel_size
                    b_load_offset = weight_offset + (n + tl.arange(0, BLOCK_SIZE_K))
                    b_tile = tl.load(weight_ptr + b_load_offset, mask=b_load_mask)
                    
                    # Compute dot product
                    acc = tl.dot(a_tile, b_tile, acc)
        
        # Add bias and store result
        output_offset = batch_id * out_channels * output_length + out_channel_id * output_length
        output_base = output_ptr + output_offset
        
        for i in range(BLOCK_SIZE_M):
            if i < output_length:
                result = acc[i, 0] + bias_val
                tl.store(output_base + i, result)

# Simplified fused approach using direct computation
@triton.jit
def conv1d_transpose_simple_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    groups,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program processes one output position
    output_pos = pid
    
    if output_pos >= output_length:
        return
        
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Load bias if available
    bias_val = 0.0
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + output_pos % out_channels)
    
    # Compute convolution
    for g in range(groups):
        group_in_start = g * (in_channels // groups)
        group_out_start = g * (out_channels // groups)
        
        # For each input position that contributes to this output position
        for k in range(kernel_size):
            # Map output position back to input position
            input_pos = (output_pos + padding - k) // stride
            
            if input_pos >= 0 and input_pos < input_length:
                # For each input channel in this group
                for c in range(in_channels // groups):
                    input_idx = batch_size * in_channels * input_length + \
                               (group_in_start + c) * input_length + input_pos
                    weight_idx = (group_in_start + c) * out_channels + \
                                (output_pos % out_channels) * kernel_size + k
                    
                    input_val = tl.load(input_ptr + input_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    acc += input_val * weight_val
    
    # Store result with bias
    output_idx = batch_size * out_channels * output_length + \
                 (output_pos % out_channels) * output_length + output_pos
    tl.store(output_ptr + output_idx, acc[0] + bias_val)

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
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        batch_size, _, input_length = x.shape
        
        # Calculate output length
        output_length = (input_length - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        # Ensure tensors are contiguous and on CUDA
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda() if self.bias is not None else None
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, output_length, dtype=torch.float32, device='cuda')
        
        # Define grid configuration
        grid = (output_length, self.out_channels, self.groups)
        
        # Launch kernel
        conv1d_transpose_simple_kernel[grid](
            x, weight, bias, output,
            batch_size, self.in_channels, self.out_channels,
            input_length, output_length,
            self.kernel_size, self.stride, self.padding,
            self.groups, BLOCK_SIZE=128
        )
        
        return output

# For testing purposes, we can still use the original implementation as fallback
class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d_transpose(x)