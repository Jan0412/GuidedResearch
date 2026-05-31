import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def maxpool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and channel index
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting position for this block
    output_row_start = tl.program_id(2) * BLOCK_SIZE
    output_col_start = tl.program_id(3) * BLOCK_SIZE
    
    # Shared memory for reduction
    shared_max = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over the kernel
    for kh in range(0, kernel_h):
        for kw in range(0, kernel_w):
            # Calculate input positions
            input_row = output_row_start * stride_h - padding_h + kh * dilation_h
            input_col = output_col_start * stride_w - padding_w + kw * dilation_w
            
            # Check if we're within bounds
            valid_row = (input_row >= 0) & (input_row < input_height)
            valid_col = (input_col >= 0) & (input_col < input_width)
            
            # Load from global memory
            if valid_row & valid_col:
                # Calculate global indices
                input_idx = (
                    batch_idx * (channels * input_height * input_width) +
                    channel_idx * (input_height * input_width) +
                    input_row * input_width + input_col
                )
                
                # Load input value
                input_val = tl.load(input_ptr + input_idx, mask=True)
                
                # Store in shared memory
                shared_max[output_row_start % BLOCK_SIZE, output_col_start % BLOCK_SIZE] = input_val
    
    # Reduce within block
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if i < output_height and j < output_width:
                # Compute maximum for this output location
                current_max = shared_max[i, j]
                # We'll do a simple reduction approach here
                # In practice, you'd want a more sophisticated reduction
                # For now, we'll just compute the max manually
                pass

@triton.jit
def fused_maxpool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_row = tl.program_id(2) * BLOCK_SIZE_H
    output_col = tl.program_id(3) * BLOCK_SIZE_W
    
    # Loop over the kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input coordinates
            input_row = output_row * stride_h - padding_h + kh * dilation_h
            input_col = output_col * stride_w - padding_w + kw * dilation_w
            
            # Bounds checking
            valid = (input_row >= 0) & (input_row < input_height) & \
                   (input_col >= 0) & (input_col < input_width)
            
            if valid:
                # Calculate input index
                input_idx = (
                    batch_idx * (channels * input_height * input_width) +
                    channel_idx * (input_height * input_width) +
                    input_row * input_width + input_col
                )
                
                # Load input value
                input_val = tl.load(input_ptr + input_idx, mask=True)
                
                # Store in output buffer (this needs to be done carefully for max operations)
                # Since we're processing multiple kernel elements, we need to properly 
                # handle the reduction operation - this is simplified for demonstration
                pass

# Optimized version using proper reduction logic
@triton.jit
def efficient_maxpool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_row = tl.program_id(2)
    output_col = tl.program_id(3)
    
    # Calculate starting input coordinates
    start_input_row = output_row * stride_h - padding_h
    start_input_col = output_col * stride_w - padding_w
    
    # Initialize maximum value
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    
    # Iterate through kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate actual input coordinates
            input_row = start_input_row + kh * dilation_h
            input_col = start_input_col + kw * dilation_w
            
            # Bounds checking
            valid = (input_row >= 0) & (input_row < input_height) & \
                   (input_col >= 0) & (input_col < input_width)
            
            # Calculate input index
            if valid:
                input_idx = (
                    batch_idx * (channels * input_height * input_width) +
                    channel_idx * (input_height * input_width) +
                    input_row * input_width + input_col
                )
                
                # Load input value and update maximum
                input_val = tl.load(input_ptr + input_idx, mask=True)
                max_val = tl.maximum(max_val, input_val)
    
    # Write result
    output_idx = (
        batch_idx * (channels * output_height * output_width) +
        channel_idx * (output_height * output_width) +
        output_row * output_width + output_col
    )
    
    tl.store(output_ptr + output_idx, max_val[0])

def triton_maxpool2d(input_tensor, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h=1, dilation_w=1):
    """
    Apply 2D max pooling using Triton kernel
    """
    batch_size, channels, input_height, input_width = input_tensor.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    output_width = (input_width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, dtype=torch.float32, device=input_tensor.device)
    
    # Grid configuration
    grid = (
        batch_size,
        channels,
        (output_height + 15) // 16,  # Block size of 16
        (output_width + 15) // 16   # Block size of 16
    )
    
    # Launch kernel
    efficient_maxpool2d_kernel[grid](
        input_tensor,
        output,
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        BLOCK_SIZE=16
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer with Triton optimization.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Max Pooling 2D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        return triton_maxpool2d(
            x, 
            self.kernel_size, 
            self.kernel_size, 
            self.stride, 
            self.stride, 
            self.padding, 
            self.padding,
            self.dilation,
            self.dilation
        )

# The rest of the original functions remain unchanged
batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]