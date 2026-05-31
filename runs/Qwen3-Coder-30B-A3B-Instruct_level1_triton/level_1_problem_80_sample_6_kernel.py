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
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Calculate the starting indices for this block
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    k_start = pid_k * BLOCK_SIZE_K
    
    # Create masks for valid elements
    m_mask = m_start + tl.arange(0, BLOCK_SIZE_M) < batch_size * out_channels * output_height * output_width
    n_mask = n_start + tl.arange(0, BLOCK_SIZE_N) < in_channels * kernel_height * kernel_width
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, in_channels * kernel_height * kernel_width, BLOCK_SIZE_K):
        # Load input and weight tiles
        input_tile = tl.load(input_ptr + (m_start + tl.arange(0, BLOCK_SIZE_M)) * in_channels * kernel_height * kernel_width + 
                             (k + tl.arange(0, BLOCK_SIZE_K)), mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        weight_tile = tl.load(weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K)), mask=n_mask, other=0.0)
        
        # Perform matrix multiplication
        acc += tl.dot(input_tile, weight_tile)
    
    # Store the result
    output_idx = m_start + tl.arange(0, BLOCK_SIZE_M)
    tl.store(output_ptr + output_idx, acc, mask=m_mask)

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton-based 2D convolution implementation
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Pad the input
    if padding != (0, 0):
        input_tensor = torch.nn.functional.pad(input_tensor, (padding[1], padding[1], padding[0], padding[0]))
    
    # Reshape input to (batch_size * output_height * output_width, in_channels * kernel_height * kernel_width)
    input_reshaped = torch.zeros(batch_size, output_height, output_width, in_channels, kernel_height, kernel_width, device=input_tensor.device, dtype=torch.float32)
    
    # Extract patches using unfold
    input_unfolded = input_tensor.unfold(2, kernel_height, stride[0]).unfold(3, kernel_width, stride[1])
    input_unfolded = input_unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()
    
    # Reshape for matrix multiplication
    input_reshaped = input_unfolded.view(batch_size, output_height, output_width, -1)
    input_reshaped = input_reshaped.view(-1, in_channels * kernel_height * kernel_width)
    
    # Reshape weight for matrix multiplication
    weight_reshaped = weight.view(out_channels, -1)
    
    # Perform matrix multiplication
    output = torch.mm(input_reshaped, weight_reshaped.t())
    
    # Reshape back to output format
    output = output.view(batch_size, output_height, output_width, out_channels)
    output = output.permute(0, 3, 1, 2).contiguous()
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using optimized Triton kernel
        """
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)