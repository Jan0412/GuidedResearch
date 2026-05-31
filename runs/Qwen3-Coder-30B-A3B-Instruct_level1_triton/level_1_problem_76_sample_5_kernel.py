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
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate base pointers for this thread's work
    input_base = input_ptr + batch_idx * in_channels * input_length
    weight_base = weight_ptr + channel_idx * out_channels * kernel_size
    output_base = output_ptr + batch_idx * out_channels * output_length
    
    # Process multiple channels if needed
    for c in range(CHANNELS_PER_BLOCK):
        channel = channel_idx * CHANNELS_PER_BLOCK + c
        if channel >= in_channels:
            break
            
        # Process multiple outputs if needed
        for o in range(OUTPUTS_PER_BLOCK):
            out_pos = output_idx * OUTPUTS_PER_BLOCK + o
            if out_pos >= output_length:
                break
                
            # Compute convolution for this output position
            acc = tl.zeros((1,), dtype=tl.float32)
            
            # Apply kernel
            for k in range(kernel_size):
                # Calculate input position with stride and dilation
                input_pos = out_pos * stride + k * dilation
                
                # Check bounds
                if input_pos >= 0 and input_pos < input_length:
                    # Load input value
                    input_val = tl.load(input_base + channel * input_length + input_pos, mask=True)
                    
                    # Load weight
                    weight_val = tl.load(weight_base + channel * out_channels * kernel_size + 
                                       k * out_channels + out_pos % out_channels, mask=True)
                    
                    acc += input_val * weight_val
            
            # Add bias if present
            if bias_ptr is not None:
                bias_val = tl.load(bias_ptr + out_pos % out_channels, mask=True)
                acc += bias_val
                
            # Store result
            tl.store(output_base + out_pos * out_channels + out_pos % out_channels, acc, mask=True)

def triton_conv1d(input_tensor, weight, bias, stride, dilation):
    """
    Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * 0 - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous and on correct device
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Set up launch parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 4
    OUTPUTS_PER_BLOCK = 4
    
    # Grid configuration
    grid = (
        batch_size,  # batch dimension
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,  # channel dimension
        (output_length + OUTPUTS_PER_BLOCK - 1) // OUTPUTS_PER_BLOCK   # output dimension
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
        dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUTS_PER_BLOCK=OUTPUTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, 
    potentially dilated and strided, optimized with Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
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
        Performs the 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)