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
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the block ID for the current thread
    pid = tl.program_id(0)
    
    # Calculate which output channel this block handles
    output_channel_id = pid % out_channels
    
    # Calculate which batch this block handles
    batch_id = pid // out_channels
    
    # Each block processes one output channel
    # Calculate the output position for this block
    output_pos = batch_id * out_channels * output_length + output_channel_id * output_length
    
    # Shared memory for input window
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Process the output along the sequence dimension
    for output_idx in range(0, output_length, BLOCK_SIZE):
        # Load input data into shared memory
        input_offset = batch_id * in_channels * input_length + output_channel_id * input_length
        input_load_idx = output_idx + tl.arange(0, BLOCK_SIZE)
        mask = input_load_idx < output_length
        
        # Load weights for this channel and kernel position
        weight_offset = output_channel_id * in_channels * kernel_size
        weight_load_idx = tl.arange(0, kernel_size)
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # For each input channel
        for ic in range(in_channels):
            # Load input data (with proper indexing)
            input_data = tl.load(input_ptr + input_offset + ic * input_length + input_load_idx, mask=mask, other=0.0)
            
            # For each kernel element
            for k in range(kernel_size):
                # Compute the actual input index considering stride, dilation, and padding
                input_index = (output_idx + k * dilation - padding) // stride
                
                # Check if input index is valid
                valid_mask = (input_index >= 0) & (input_index < input_length) & (output_idx + k * dilation - padding) % stride == 0
                
                # Load weight
                weight_val = tl.load(weight_ptr + weight_offset + ic * kernel_size + k, mask=valid_mask, other=0.0)
                
                # Accumulate
                acc += input_data * weight_val
            
            # Add bias if it exists
            if bias_ptr is not None:
                bias_val = tl.load(bias_ptr + output_channel_id, mask=True)
                acc += bias_val
        
        # Store results
        output_load_idx = output_idx + tl.arange(0, BLOCK_SIZE)
        output_mask = output_load_idx < output_length
        tl.store(output_ptr + output_pos + output_load_idx, acc, mask=output_mask)

def triton_conv1d_transpose(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
):
    """
    Triton implementation of Conv1dTranspose
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Launch kernel
    grid_size = batch_size * out_channels
    BLOCK_SIZE = 128
    
    # Configure grid
    grid = lambda meta: (grid_size,)
    
    # Call kernel
    conv1d_transpose_kernel[grid](
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
        dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_M=8
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Use our Triton implementation instead of PyTorch's native implementation
        return triton_conv1d_transpose(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation
        )

# Note: This implementation is simplified for demonstration purposes.
# A full production version would require more careful handling of boundary conditions,
# memory coalescing, and edge case optimizations.