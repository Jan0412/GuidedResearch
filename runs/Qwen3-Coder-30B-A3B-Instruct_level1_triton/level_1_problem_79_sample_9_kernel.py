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
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_LENGTH_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_pos_id = tl.program_id(2)
    
    # Calculate output position
    output_pos_start = output_pos_id * OUTPUT_LENGTH_PER_BLOCK
    output_pos_end = min(output_pos_start + OUTPUT_LENGTH_PER_BLOCK, output_length)
    
    # Shared memory for input chunk
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(OUTPUT_LENGTH_PER_BLOCK,))
    
    # Process multiple channels per thread block
    for c in range(channel_id * CHANNELS_PER_BLOCK, min((channel_id + 1) * CHANNELS_PER_BLOCK, out_channels)):
        # Initialize accumulator
        acc = tl.zeros((OUTPUT_LENGTH_PER_BLOCK,), dtype=tl.float32)
        
        # Loop over kernel positions
        for k in range(kernel_size):
            # Calculate input position for this kernel element
            input_pos = output_pos_start * stride - padding + k * dilation
            
            # Load input chunk
            for i in range(OUTPUT_LENGTH_PER_BLOCK):
                if input_pos + i >= 0 and input_pos + i < input_length:
                    input_val = tl.load(input_ptr + 
                                      batch_id * in_channels * input_length +
                                      c * input_length +
                                      input_pos + i,
                                      mask=(input_pos + i) < input_length,
                                      other=0.0)
                    acc[i] += input_val * tl.load(weight_ptr + 
                                                c * out_channels * kernel_size +
                                                (out_channels - 1 - c) * kernel_size +
                                                k)
                else:
                    acc[i] += 0.0
        
        # Apply bias if available
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + c, mask=c < out_channels, other=0.0)
            for i in range(OUTPUT_LENGTH_PER_BLOCK):
                acc[i] += bias_val
        
        # Store output
        for i in range(OUTPUT_LENGTH_PER_BLOCK):
            if output_pos_start + i < output_length:
                tl.store(output_ptr + 
                        batch_id * out_channels * output_length +
                        c * output_length +
                        output_pos_start + i,
                        acc[i],
                        mask=output_pos_start + i < output_length)

def triton_conv1d_transpose(input_tensor, weight, bias, stride, padding, dilation):
    """
    Triton implementation of 1D transposed convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Handle bias
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Define block sizes
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 4
    OUTPUT_LENGTH_PER_BLOCK = 64
    
    # Grid dimensions
    grid = (
        batch_size,
        (out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
        (output_length + OUTPUT_LENGTH_PER_BLOCK - 1) // OUTPUT_LENGTH_PER_BLOCK
    )
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
        input_tensor.data_ptr(),
        weight.data_ptr(),
        output.data_ptr(),
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
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_LENGTH_PER_BLOCK=OUTPUT_LENGTH_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Convert to float32 for consistent computation
        if x.dtype != torch.float32:
            x = x.float()
            
        # Use Triton kernel for computation
        return triton_conv1d_transpose(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias is not None}'