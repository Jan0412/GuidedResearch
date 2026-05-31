import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_width,
    input_height,
    output_depth,
    output_width,
    output_height,
    kernel_depth,
    kernel_width,
    kernel_height,
    stride_d,
    stride_w,
    stride_h,
    padding_d,
    padding_w,
    padding_h,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_d_id = tl.program_id(2)
    
    # Calculate output dimensions
    output_elements = output_depth * output_width * output_height
    
    # Shared memory for output block
    output_block = tl.shared_memory(dtype=tl.float32, size=OUTPUT_BLOCK_SIZE)
    
    # Process output elements in chunks
    for i in range((output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE):
        # Calculate global output index
        global_idx = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        # Filter valid indices
        valid = global_idx < output_elements
        
        # Convert linear index to 3D coordinates
        out_h = global_idx % output_height
        out_w = (global_idx // output_height) % output_width
        out_d = (global_idx // (output_height * output_width)) % output_depth
        
        # Apply padding offset
        out_d_padded = out_d - padding_d
        out_w_padded = out_w - padding_w
        out_h_padded = out_h - padding_h
        
        # Check if this position is valid for convolution
        valid_conv = valid & (out_d_padded >= 0) & (out_w_padded >= 0) & (out_h_padded >= 0)
        valid_conv = valid_conv & (out_d_padded < input_depth) & (out_w_padded < input_width) & (out_h_padded < input_height)
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Loop over input channels
        for ic in range(in_channels):
            # For each kernel position
            for kd in range(kernel_depth):
                for kw in range(kernel_width):
                    for kh in range(kernel_height):
                        # Calculate input position
                        in_d = out_d_padded - kd
                        in_w = out_w_padded - kw
                        in_h = out_h_padded - kh
                        
                        # Check bounds for input
                        valid_input = (in_d >= 0) & (in_w >= 0) & (in_h >= 0) & \
                                      (in_d < input_depth) & (in_w < input_width) & (in_h < input_height)
                        
                        # Compute weights
                        weight_idx = out_ch_id * in_channels * kernel_depth * kernel_width * kernel_height + \
                                    ic * kernel_depth * kernel_width * kernel_height + \
                                    kd * kernel_width * kernel_height + \
                                    kw * kernel_height + kh
                        
                        # Compute input value
                        input_idx = batch_id * in_channels * input_depth * input_width * input_height + \
                                   ic * input_depth * input_width * input_height + \
                                   in_d * input_width * input_height + \
                                   in_w * input_height + in_h
                        
                        # Load weight and input
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        input_val = tl.load(input_ptr + input_idx, mask=valid_input, other=0.0)
                        
                        # Accumulate
                        acc += weight_val * input_val
        
        # Store results
        out_idx = batch_id * out_channels * output_depth * output_width * output_height + \
                 out_ch_id * output_depth * output_width * output_height + \
                 out_d * output_width * output_height + \
                 out_w * output_height + out_h
        
        # Only store valid outputs
        tl.store(output_ptr + out_idx, acc[0], mask=valid[0])

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0)):
    """
    Custom Triton implementation of ConvTranspose3d
    """
    # Get dimensions
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_w, stride_h = stride
    pad_d, pad_w, pad_h = padding
    
    output_depth = (input_depth - 1) * stride_d - 2 * pad_d + kernel_depth
    output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width
    output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_width, output_height, device=input_tensor.device, dtype=torch.float32)
    
    # Prepare parameters
    BLOCK_SIZE = 32
    OUTPUT_BLOCK_SIZE = 256
    
    # Grid configuration
    grid = (
        batch_size,           # batch dimension
        out_channels,         # output channel dimension  
        output_depth          # output depth dimension
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_width,
        input_height,
        output_depth,
        output_width,
        output_height,
        kernel_depth,
        kernel_width,
        kernel_height,
        stride_d,
        stride_w,
        stride_h,
        pad_d,
        pad_w,
        pad_h,
        BLOCK_SIZE,
        OUTPUT_BLOCK_SIZE
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Ensure proper initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding
        )

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, output_padding={self.output_padding}, groups={self.groups}, bias={self.bias}'