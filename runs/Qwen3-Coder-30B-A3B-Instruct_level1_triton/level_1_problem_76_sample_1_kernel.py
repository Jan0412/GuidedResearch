import torch
import torch.nn as nn
import torch.nn.functional as F
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
    dilation,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_id = tl.program_id(2)
    
    # Shared memory for input window
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Calculate global indices
    batch_offset = batch_id * in_channels * input_length
    input_offset = batch_offset + channel_id * input_length
    weight_offset = channel_id * out_channels * kernel_size
    output_offset = batch_id * out_channels * output_length + channel_id * output_length
    
    # Process output elements in chunks
    for out_idx in range(output_id * OUTPUTS_PER_BLOCK, min((output_id + 1) * OUTPUTS_PER_BLOCK, output_length)):
        if out_idx >= output_length:
            break
            
        # Compute the starting position in the input for this output
        start_pos = out_idx * stride
        
        # Initialize accumulator
        acc = tl.zeros((out_channels,), dtype=tl.float32)
        
        # For each output channel
        for oc in range(out_channels):
            # Compute convolution for this output channel
            temp_acc = 0.0
            
            # For each kernel element
            for k in range(kernel_size):
                # Calculate input position
                input_pos = start_pos + k * dilation
                
                # Check bounds
                if input_pos >= 0 and input_pos < input_length:
                    # Load input value
                    input_val = tl.load(input_ptr + input_offset + input_pos, mask=(input_pos < input_length))
                    
                    # Load weight
                    weight_val = tl.load(weight_ptr + weight_offset + oc * kernel_size + k)
                    
                    # Accumulate
                    temp_acc += input_val * weight_val
                else:
                    # Out of bounds, contribution is zero
                    pass
            
            # Add bias if present
            if bias_ptr is not None:
                bias_val = tl.load(bias_ptr + oc)
                temp_acc += bias_val
                
            acc[oc] = temp_acc
        
        # Store output
        for oc in range(out_channels):
            if oc < out_channels:
                tl.store(output_ptr + output_offset + out_idx * out_channels + oc, acc[oc])

@triton.jit
def conv1d_fused_kernel(
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
    dilation,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_id = tl.program_id(2)
    
    # Shared memory for input window
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Calculate global indices
    batch_offset = batch_id * in_channels * input_length
    input_offset = batch_offset + channel_id * input_length
    weight_offset = channel_id * out_channels * kernel_size
    output_offset = batch_id * out_channels * output_length + channel_id * output_length
    
    # Process output elements in chunks
    for out_idx in range(output_id * OUTPUTS_PER_BLOCK, min((output_id + 1) * OUTPUTS_PER_BLOCK, output_length)):
        if out_idx >= output_length:
            break
            
        # Compute the starting position in the input for this output
        start_pos = out_idx * stride
        
        # Initialize accumulator
        acc = tl.zeros((out_channels,), dtype=tl.float32)
        
        # For each output channel
        for oc in range(out_channels):
            # Compute convolution for this output channel
            temp_acc = 0.0
            
            # For each kernel element
            for k in range(kernel_size):
                # Calculate input position
                input_pos = start_pos + k * dilation
                
                # Check bounds
                if input_pos >= 0 and input_pos < input_length:
                    # Load input value
                    input_val = tl.load(input_ptr + input_offset + input_pos, mask=(input_pos < input_length))
                    
                    # Load weight
                    weight_val = tl.load(weight_ptr + weight_offset + oc * kernel_size + k)
                    
                    # Accumulate
                    temp_acc += input_val * weight_val
                else:
                    # Out of bounds, contribution is zero
                    pass
            
            # Add bias if present
            if bias_ptr is not None:
                bias_val = tl.load(bias_ptr + oc)
                temp_acc += bias_val
                
            acc[oc] = temp_acc
        
        # Store output
        for oc in range(out_channels):
            if oc < out_channels:
                tl.store(output_ptr + output_offset + out_idx * out_channels + oc, acc[oc])

def triton_conv1d(input_tensor, weight, bias, stride, dilation):
    """
    Triton implementation of 1D convolution with fused operations
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * 0 - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Define grid dimensions
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 1
    OUTPUTS_PER_BLOCK = 8
    
    # Grid configuration
    grid = (
        batch_size,
        in_channels,
        (output_length + OUTPUTS_PER_BLOCK - 1) // OUTPUTS_PER_BLOCK
    )
    
    # Launch kernel
    conv1d_fused_kernel[grid](
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
        dilation,
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK,
        OUTPUTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized version using Triton kernels for 1D convolution
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel
        """
        # Use our optimized Triton implementation
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)