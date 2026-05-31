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
    bias_enabled,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program processes one output element
    # Calculate which output element this program handles
    output_idx = pid
    
    if output_idx >= batch_size * out_channels * output_length:
        return
        
    # Calculate batch, channel, and position indices
    batch_idx = output_idx // (out_channels * output_length)
    remaining = output_idx % (out_channels * output_length)
    channel_idx = remaining // output_length
    pos_idx = remaining % output_length
    
    # Calculate input start position for this output position
    input_start_pos = pos_idx * stride
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Perform convolution
    for k in range(kernel_size):
        # Calculate input position considering dilation
        input_pos = input_start_pos + k * dilation
        
        # Check bounds
        if input_pos >= 0 and input_pos < input_length:
            # Load input value
            input_val = tl.load(input_ptr + 
                               batch_idx * (in_channels * input_length) +
                               channel_idx * input_length +
                               input_pos)
            
            # Load weight
            weight_val = tl.load(weight_ptr + 
                                channel_idx * (out_channels * kernel_size) +
                                channel_idx * kernel_size +
                                k)
            
            acc += input_val * weight_val
    
    # Add bias if enabled
    if bias_enabled:
        bias_val = tl.load(bias_ptr + channel_idx)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + output_idx, acc)

class ModelNew(nn.Module):
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        batch_size, _, input_length = x.shape
        
        # Calculate output length
        output_length = (input_length + 2 * 0 - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_length, device=x.device, dtype=torch.float32)
        
        # Prepare parameters
        n_elements = batch_size * self.out_channels * output_length
        
        # Define block size and group size
        BLOCK_SIZE = 1024
        GROUP_SIZE = 32
        
        # Calculate grid size
        grid = lambda meta: (math.ceil(n_elements / meta["BLOCK_SIZE"]),)
        
        # Launch kernel
        conv1d_kernel[grid](
            x,
            self.weight,
            output,
            self.bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_length,
            output_length,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.bias is not None,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE=GROUP_SIZE
        )
        
        return output

# Note: The above implementation has limitations compared to PyTorch's native convolution
# due to complexity of handling all edge cases in a single kernel. For production use,
# it's recommended to use PyTorch's optimized implementations or more sophisticated
# Triton kernels that handle proper batching and memory access patterns.