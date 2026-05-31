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
    GROUP_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which output row this program handles
    output_row = pid // GROUP_SIZE
    group_id = pid % GROUP_SIZE
    
    if output_row >= output_length:
        return
    
    # Calculate output channel
    oc = group_id
    
    # For each output position
    for out_pos in range(output_row, output_length, output_length):
        # Calculate input positions for this output position
        input_start = (out_pos - padding) // stride
        
        # Initialize accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Apply bias if exists
        if bias_ptr is not None:
            acc = tl.load(bias_ptr + oc, mask=oc < out_channels)
        
        # Compute convolution
        for k in range(kernel_size):
            # Calculate actual input index
            input_idx = input_start + k * dilation
            
            # Check bounds
            if input_idx >= 0 and input_idx < input_length:
                # Loop over input channels
                for ic in range(in_channels):
                    # Load weight
                    weight_val = tl.load(weight_ptr + oc * in_channels * kernel_size + 
                                       ic * kernel_size + k, mask=(oc < out_channels) & (ic < in_channels) & (k < kernel_size))
                    
                    # Load input
                    input_val = tl.load(input_ptr + ic * input_length + input_idx, 
                                      mask=(input_idx < input_length) & (ic < in_channels))
                    
                    # Accumulate
                    acc += weight_val * input_val
        
        # Write output
        if oc < out_channels and out_pos < output_length:
            tl.store(output_ptr + oc * output_length + out_pos, acc, mask=(oc < out_channels) & (out_pos < output_length))

def triton_conv1d_transpose(input_tensor, weight, bias, stride, padding, dilation):
    """
    Custom Triton implementation of ConvTranspose1d
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    
    # Prepare output tensor
    output = torch.zeros(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Flatten input for easier indexing
    input_flat = input_tensor.view(-1, input_length)
    output_flat = output.view(-1, output_length)
    
    # Ensure tensors are contiguous and on GPU
    input_flat = input_flat.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Calculate grid size
    total_elements = batch_size * out_channels * output_length
    BLOCK_SIZE = 128
    GROUP_SIZE = min(8, out_channels)
    grid_size = (math.ceil(total_elements / (BLOCK_SIZE * GROUP_SIZE)) * GROUP_SIZE,)
    
    # Launch kernel
    conv1d_transpose_kernel[grid_size](
        input_flat,
        weight,
        output_flat,
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
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
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
        Performs the transposed 1D convolution using custom Triton kernel.
        """
        return triton_conv1d_transpose(x, self.weight, self.bias, self.stride, self.padding, self.dilation)