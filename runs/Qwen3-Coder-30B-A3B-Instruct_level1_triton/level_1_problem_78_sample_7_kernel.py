import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program processes one output channel
    output_channel = pid
    
    if output_channel >= out_channels:
        return
        
    # Calculate output dimensions
    # For transposed conv, output size = (input_size - 1) * stride - 2 * padding + kernel_size
    
    # Each program handles a chunk of the output
    output_elements_per_program = (output_height * output_width + GROUP_SIZE - 1) // GROUP_SIZE
    start_element = pid * output_elements_per_program
    end_element = min(start_element + output_elements_per_program, output_height * output_width)
    
    # Loop over batches
    for batch in range(batch_size):
        # Loop over input channels
        for ic in range(in_channels):
            # Loop over output spatial locations
            for oh in range(output_height):
                for ow in range(output_width):
                    # Calculate the corresponding input position
                    ih = oh // stride_h
                    iw = ow // stride_w
                    
                    # Check if this output position contributes to the current input position
                    if (oh - ih * stride_h) == 0 and (ow - iw * stride_w) == 0:
                        # Compute convolution
                        acc = 0.0
                        
                        # Loop over kernel
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                # Calculate input indices
                                ih_idx = ih - pad_h + kh
                                iw_idx = iw - pad_w + kw
                                
                                # Check bounds
                                if ih_idx >= 0 and ih_idx < input_height and iw_idx >= 0 and iw_idx < input_width:
                                    # Get input value
                                    input_val = tl.load(input_ptr + 
                                                       batch * (in_channels * input_height * input_width) +
                                                       ic * (input_height * input_width) +
                                                       ih_idx * input_width + 
                                                       iw_idx)
                                    
                                    # Get weight value
                                    weight_val = tl.load(weight_ptr + 
                                                        output_channel * (in_channels * kernel_h * kernel_w) +
                                                        ic * (kernel_h * kernel_w) +
                                                        kh * kernel_w + 
                                                        kw)
                                    
                                    acc += input_val * weight_val
                        
                        # Apply bias if present
                        if bias_ptr is not None:
                            bias_val = tl.load(bias_ptr + output_channel)
                            acc += bias_val
                        
                        # Store result
                        tl.store(output_ptr + 
                                batch * (out_channels * output_height * output_width) +
                                output_channel * (output_height * output_width) +
                                oh * output_width + 
                                ow, 
                                acc)

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding, output_padding):
    """
    Triton implementation of ConvTranspose2d
    """
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_h
    output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_w
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, dtype=torch.float32, device=input_tensor.device)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    GROUP_SIZE = 32
    
    # Launch kernel
    num_programs = out_channels
    grid = lambda meta: (num_programs,)
    
    # Call the kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        """
        return triton_conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, (0, 0))