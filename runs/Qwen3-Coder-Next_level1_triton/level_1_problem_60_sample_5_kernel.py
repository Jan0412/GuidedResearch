import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr, output_ptr, weight_ptr, bias_ptr,
    batch_size, in_channels, out_channels,
    input_width, input_height, input_depth,
    output_width, output_height, output_depth,
    kernel_width, kernel_height, kernel_depth,
    stride, padding, dilation,
    # Strides for each dimension
    input_batch_stride, input_channel_stride,
    input_width_stride, input_height_stride, input_depth_stride,
    weight_out_channel_stride, weight_in_channel_stride,
    weight_kd_stride, weight_kh_stride, weight_kw_stride,
    output_batch_stride, output_channel_stride,
    output_width_stride, output_height_stride, output_depth_stride,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for computation (sum dimension)
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_channel = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    pid_d = tl.program_id(4)
    
    # Calculate output position
    out_h = pid_h
    out_w = pid_w
    out_d = pid_d
    
    # Calculate input position corresponding to this output
    in_w = out_w * stride - padding + pid_w * 0  # Will be adjusted in loop
    in_h = out_h * stride - padding + pid_h * 0
    in_d = out_d * stride - padding + pid_d * 0
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Convolution loop
    for ic in range(in_channels):
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input position with dilation
                    input_w = out_w * stride - padding + kw * dilation + pid_w * 0
                    input_h = out_h * stride - padding + kh * dilation + pid_h * 0
                    input_d = out_d * stride - padding + kd * dilation + pid_d * 0
                    
                    # Check bounds
                    if (input_w >= 0 and input_w < input_width and 
                        input_h >= 0 and input_h < input_height and 
                        input_d >= 0 and input_d < input_depth):
                        
                        # Calculate input pointer offset
                        input_offset = (pid_batch * input_batch_stride + 
                                       ic * input_channel_stride + 
                                       input_h * input_height_stride + 
                                       input_w * input_width_stride + 
                                       input_d * input_depth_stride)
                        
                        # Load input value
                        x = tl.load(input_ptr + input_offset)
                        
                        # Calculate weight pointer offset
                        weight_offset = (pid_out_channel * weight_out_channel_stride + 
                                        ic * weight_in_channel_stride + 
                                        kd * weight_kd_stride + 
                                        kh * weight_kh_stride + 
                                        kw * weight_kw_stride)
                        
                        # Load weight value
                        w = tl.load(weight_ptr + weight_offset)
                        
                        # Accumulate
                        acc += x * w
    
    # Add bias if available
    if bias_ptr is not None:
        bias_offset = pid_out_channel
        acc += tl.load(bias_ptr + bias_offset)
    
    # Store result
    output_offset = (pid_batch * output_batch_stride + 
                    pid_out_channel * output_channel_stride + 
                    out_h * output_height_stride + 
                    out_w * output_width_stride + 
                    out_d * output_depth_stride)
    
    tl.store(output_ptr + output_offset, acc.to(tl.float32))


def triton_conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 3D convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, input_width, input_height, input_depth = x.shape
    out_channels, _, kernel_width, kernel_height, kernel_depth = weight.shape
    
    # Calculate output dimensions
    output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    output_depth = (input_depth + 2 * padding - dilation * (kernel_depth - 1) - 1) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_width, output_height, output_depth, 
                        dtype=x.dtype, device=x.device)
    
    # Calculate strides
    input_batch_stride = in_channels * input_width * input_height * input_depth
    input_channel_stride = input_width * input_height * input_depth
    input_width_stride = input_height * input_depth
    input_height_stride = input_depth
    input_depth_stride = 1
    
    weight_out_channel_stride = in_channels * kernel_width * kernel_height * kernel_depth
    weight_in_channel_stride = kernel_width * kernel_height * kernel_depth
    weight_kd_stride = kernel_height * kernel_width
    weight_kh_stride = kernel_width
    weight_kw_stride = 1
    
    output_batch_stride = out_channels * output_width * output_height * output_depth
    output_channel_stride = output_width * output_height * output_depth
    output_width_stride = output_height * output_depth
    output_height_stride = output_depth
    output_depth_stride = 1
    
    # Grid dimensions
    grid = (batch_size, out_channels, output_height, output_width, output_depth)
    
    # Launch kernel
    conv3d_kernel[grid](
        x, output, weight, bias,
        batch_size, in_channels, out_channels,
        input_width, input_height, input_depth,
        output_width, output_height, output_depth,
        kernel_width, kernel_height, kernel_depth,
        stride, padding, dilation,
        input_batch_stride, input_channel_stride,
        input_width_stride, input_height_stride, input_depth_stride,
        weight_out_channel_stride, weight_in_channel_stride,
        weight_kd_stride, weight_kh_stride, weight_kw_stride,
        output_batch_stride, output_channel_stride,
        output_width_stride, output_height_stride, output_depth_stride,
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=1, BLOCK_SIZE_K=1
    )
    
    return output


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.
    Optimized with Triton kernels for improved performance.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using optimized Triton kernel.
        """
        # Get parameters from the original conv3d layer
        weight = self.conv3d.weight
        bias = self.conv3d.bias
        
        # Use our optimized Triton convolution
        return triton_conv3d(x, weight, bias, 
                            stride=self.conv3d.stride[0],
                            padding=self.conv3d.padding[0],
                            dilation=self.conv3d.dilation[0],
                            groups=self.conv3d.groups)