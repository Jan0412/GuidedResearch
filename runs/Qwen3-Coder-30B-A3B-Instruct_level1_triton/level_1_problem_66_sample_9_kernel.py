import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_d,
    kernel_h,
    kernel_w,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    dilation_d,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate output dimensions
    total_output_elements = batch_size * out_channels * output_depth * output_height * output_width
    
    # Each thread handles one output element
    output_element_idx = output_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_element_idx < total_output_elements
    
    if not mask[0]:
        return
        
    # Decompose linear index into multi-dimensional indices
    temp = output_element_idx
    out_w = temp % output_width
    temp = temp // output_width
    out_h = temp % output_height
    temp = temp // output_height
    out_d = temp % output_depth
    temp = temp // output_depth
    out_c = temp % out_channels
    temp = temp // out_channels
    batch = temp
    
    # Calculate corresponding input positions
    input_d_start = out_d * stride_d - padding_d
    input_h_start = out_h * stride_h - padding_h
    input_w_start = out_w * stride_w - padding_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Perform convolution
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input coordinates
                input_d = input_d_start + kd * dilation_d
                input_h = input_h_start + kh * dilation_h
                input_w = input_w_start + kw * dilation_w
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and 
                    input_h >= 0 and input_h < input_height and 
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input and weight indices
                    input_idx = (
                        batch * (in_channels * input_depth * input_height * input_width) +
                        channel_idx * (input_depth * input_height * input_width) +
                        input_d * (input_height * input_width) +
                        input_h * input_width +
                        input_w
                    )
                    
                    weight_idx = (
                        out_c * (in_channels * kernel_d * kernel_h * kernel_w) +
                        channel_idx * (kernel_d * kernel_h * kernel_w) +
                        kd * (kernel_h * kernel_w) +
                        kh * kernel_w +
                        kw
                    )
                    
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    acc += input_val * weight_val
    
    # Store result
    output_idx_final = (
        batch * (out_channels * output_depth * output_height * output_width) +
        out_c * (output_depth * output_height * output_width) +
        out_d * (output_height * output_width) +
        out_h * output_width +
        out_w
    )
    
    tl.store(output_ptr + output_idx_final, acc, mask=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # For simplicity, we'll use PyTorch's implementation for initialization
        # but replace the forward pass with Triton kernel
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize weights using PyTorch's default method
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using a custom Triton kernel.
        """
        batch_size, _, input_depth, input_height, input_width = x.shape
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        dilation_d, dilation_h, dilation_w = self.dilation
        
        # Calculate output dimensions
        output_depth = (input_depth + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) // stride_d + 1
        output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        
        # Ensure all tensors are on CUDA
        x = x.cuda()
        weight = self.weight.cuda()
        if self.bias is not None:
            bias = self.bias.cuda()
        
        # Prepare output tensor
        output = torch.empty(
            batch_size, self.out_channels, output_depth, output_height, output_width,
            dtype=torch.float32, device=x.device
        )
        
        # Define block size and launch parameters
        BLOCK_SIZE = 128
        CHANNELS_PER_BLOCK = 1
        OUTPUT_ELEMENTS_PER_BLOCK = 1
        
        # Grid dimensions
        grid_batch = batch_size
        grid_channels = (self.out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK
        grid_outputs = (output_depth * output_height * output_width + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        conv3d_kernel[(grid_batch, grid_channels, grid_outputs)](
            x, weight, output,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            dilation_d,
            dilation_h,
            dilation_w,
            BLOCK_SIZE=BLOCK_SIZE,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
            OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
        )
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
            
        return output