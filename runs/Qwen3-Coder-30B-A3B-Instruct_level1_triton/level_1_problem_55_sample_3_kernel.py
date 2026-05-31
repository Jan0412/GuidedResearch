import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,   # Input tensor pointer
    weight_ptr,  # Weight tensor pointer
    output_ptr,  # Output tensor pointer
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_oc, weight_stride_ic, weight_stride_h, weight_stride_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    batch_size, in_channels, out_channels, 
    input_height, input_width, 
    output_height, output_width,
    kernel_height, kernel_width,
    padding_h, padding_w,
    stride_h, stride_w,
    dilation_h, dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the block index
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Calculate the starting positions for this block
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    k_start = pid_k * BLOCK_SIZE_K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension (input channels and kernel elements)
    for k in range(0, in_channels * kernel_height * kernel_width, BLOCK_SIZE_K):
        # Load input tile
        input_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        # Load weight tile
        weight_tile = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Compute actual indices for input
        input_idx_h = m_start // output_width
        input_idx_w = m_start % output_width
        
        # Compute actual indices for output
        output_idx_h = n_start // output_width
        output_idx_w = n_start % output_width
        
        # For each kernel element
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                for ic in range(in_channels):
                    if k + ic * kernel_height * kernel_width + kh * kernel_width + kw < in_channels * kernel_height * kernel_width:
                        # Calculate input coordinates with padding
                        ih = input_idx_h * stride_h + kh * dilation_h - padding_h
                        iw = input_idx_w * stride_w + kw * dilation_w - padding_w
                        
                        # Check bounds
                        if 0 <= ih < input_height and 0 <= iw < input_width:
                            input_val = tl.load(input_ptr + 
                                              (k_start + ic * kernel_height * kernel_width + kh * kernel_width + kw) * 
                                              input_stride_c + 
                                              ih * input_stride_h + 
                                              iw * input_stride_w)
                            input_tile[0, k + ic * kernel_height * kernel_width + kh * kernel_width + kw] = input_val
                        
                        # Load weight value
                        weight_val = tl.load(weight_ptr + 
                                           (k_start + ic * kernel_height * kernel_width + kh * kernel_width + kw) * 
                                           weight_stride_ic + 
                                           (output_idx_h * output_width + output_idx_w) * weight_stride_oc)
                        weight_tile[k + ic * kernel_height * kernel_width + kh * kernel_width + kw, 0] = weight_val
                        
        # Matrix multiplication
        acc += tl.dot(input_tile, weight_tile)
    
    # Store output
    output_idx = m_start * output_stride_h + n_start * output_stride_w
    tl.store(output_ptr + output_idx, acc)

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Custom Triton implementation of 2D convolution
    """
    # Ensure inputs are contiguous and on GPU
    input_tensor = input_tensor.contiguous().cuda()
    weight = weight.contiguous().cuda()
    
    if bias is not None:
        bias = bias.contiguous().cuda()
    
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device='cuda', dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid_m = (output_height * output_width + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (in_channels * kernel_height * kernel_width + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    grid = (grid_m, grid_n, grid_k)
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_height, kernel_width,
        padding[0], padding[1],
        stride[0], stride[1],
        dilation[0], dilation[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use the custom Triton implementation instead of PyTorch's native convolution
        return triton_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            stride=self.conv2d.stride, 
            padding=self.conv2d.padding, 
            dilation=self.conv2d.dilation
        )

# Test code
batch_size = 8
# smaller spatial dims
height = 512
width = 1024
in_channels = 64  # increased channels
out_channels = 128
kernel_size = 3
# asymmetric input: make width considerably larger than height

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization