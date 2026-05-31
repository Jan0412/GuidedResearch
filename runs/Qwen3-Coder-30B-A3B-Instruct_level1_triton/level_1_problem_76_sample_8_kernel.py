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
    dilation,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate global indices
    input_base = batch_idx * in_channels * input_length
    weight_base = channel_idx * in_channels * kernel_size
    output_base = batch_idx * out_channels * output_length
    
    # Shared memory for input window
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Process in chunks of CHANNELS_PER_BLOCK
    if channel_idx * CHANNELS_PER_BLOCK >= in_channels:
        return
        
    # Initialize accumulator
    acc = tl.zeros((OUTPUTS_PER_BLOCK,), dtype=tl.float32)
    
    # For each output position
    for out_pos in range(OUTPUTS_PER_BLOCK):
        if output_idx * OUTPUTS_PER_BLOCK + out_pos >= output_length:
            break
            
        # Calculate current output position
        curr_output_pos = output_idx * OUTPUTS_PER_BLOCK + out_pos
        if curr_output_pos >= output_length:
            break
            
        # Calculate starting position in input
        start_pos = curr_output_pos * stride
        
        # Process kernel elements
        for k in range(kernel_size):
            # Calculate input position with dilation
            input_pos = start_pos + k * dilation
            
            # Load input value if within bounds
            if input_pos >= 0 and input_pos < input_length:
                # Load from shared memory or global memory
                input_val = tl.load(input_ptr + input_base + channel_idx * input_length + input_pos, mask=True)
                # Load weight
                weight_val = tl.load(weight_ptr + weight_base + k, mask=True)
                acc[out_pos] += input_val * weight_val
            else:
                # Out of bounds - skip
                pass
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + channel_idx, mask=True)
        for i in range(OUTPUTS_PER_BLOCK):
            if output_idx * OUTPUTS_PER_BLOCK + i < output_length:
                acc[i] += bias_val
    
    # Store results
    for i in range(OUTPUTS_PER_BLOCK):
        if output_idx * OUTPUTS_PER_BLOCK + i < output_length:
            tl.store(output_ptr + output_base + channel_idx * output_length + output_idx * OUTPUTS_PER_BLOCK + i, 
                    acc[i], mask=True)

def triton_conv1d(input_tensor, weight, bias, stride, dilation):
    """
    Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * 0 - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Define grid dimensions
    grid = (
        batch_size,  # batch dimension
        (in_channels + 31) // 32,  # channel dimension (rounded up)
        (output_length + 31) // 32  # output dimension (rounded up)
    )
    
    # Kernel parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 32
    OUTPUTS_PER_BLOCK = 32
    
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
        dilation,
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK,
        OUTPUTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
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
            
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)