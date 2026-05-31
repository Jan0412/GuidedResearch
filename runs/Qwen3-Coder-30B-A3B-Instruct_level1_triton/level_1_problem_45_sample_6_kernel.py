import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def avg_pool2d_kernel(
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
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Get block indices
    batch_idx = tl.program_id(0)
    channel_block = tl.program_id(1)
    
    # Calculate starting position for this block
    start_h = tl.program_id(2) * BLOCK_SIZE
    start_w = tl.program_id(3) * BLOCK_SIZE
    
    # Shared memory for reduction
    shared_mem = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Loop over kernel dimensions
    for kh in range(0, kernel_h, BLOCK_SIZE):
        for kw in range(0, kernel_w, BLOCK_SIZE):
            # Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
            count = 0
            
            # Process kernel elements
            for ih in range(kh, min(kh + BLOCK_SIZE, kernel_h)):
                for iw in range(kw, min(kw + BLOCK_SIZE, kernel_w)):
                    # Calculate input coordinates
                    ih_input = start_h * stride_h + ih - padding_h
                    iw_input = start_w * stride_w + iw - padding_w
                    
                    # Check if within bounds
                    if ih_input >= 0 and ih_input < input_height and iw_input >= 0 and iw_input < input_width:
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          batch_idx * channels * input_height * input_width +
                                          channel_block * input_height * input_width +
                                          ih_input * input_width + iw_input)
                        acc += input_val
                        count += 1
            
            # Store partial results in shared memory
            for i in range(BLOCK_SIZE):
                for j in range(BLOCK_SIZE):
                    if start_h + i < output_height and start_w + j < output_width:
                        shared_mem[i, j] = acc[i, j] if i < BLOCK_SIZE and j < BLOCK_SIZE else 0.0
    
    # Reduce within block
    if start_h < output_height and start_w < output_width:
        # Compute average
        total_count = count
        if total_count > 0:
            avg_val = shared_mem[0, 0] / total_count if start_h == 0 and start_w == 0 else 0.0
            for i in range(BLOCK_SIZE):
                for j in range(BLOCK_SIZE):
                    if start_h + i < output_height and start_w + j < output_width:
                        tl.store(output_ptr + 
                               batch_idx * channels * output_height * output_width +
                               channel_block * output_height * output_width +
                               (start_h + i) * output_width + (start_w + j),
                               avg_val)

# Optimized version using more efficient memory access patterns
@triton.jit
def avg_pool2d_kernel_optimized(
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
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)
    
    # Calculate output position
    out_h = h_idx * BLOCK_SIZE_H
    out_w = w_idx * BLOCK_SIZE_W
    
    # Shared memory for accumulating values
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate through kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input position
            ih = out_h * stride_h + kh - padding_h
            iw = out_w * stride_w + kw - padding_w
            
            # Check bounds
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_idx * channels * input_height * input_width +
                                  channel_idx * input_height * input_width +
                                  ih * input_width + iw)
                # Accumulate
                for i in range(BLOCK_SIZE_H):
                    for j in range(BLOCK_SIZE_W):
                        if out_h + i < output_height and out_w + j < output_width:
                            acc[i, j] += input_val
    
    # Compute average
    total_elements = kernel_h * kernel_w
    avg_val = acc / total_elements
    
    # Store output
    for i in range(BLOCK_SIZE_H):
        for j in range(BLOCK_SIZE_W):
            if out_h + i < output_height and out_w + j < output_width:
                tl.store(output_ptr + 
                       batch_idx * channels * output_height * output_width +
                       channel_idx * output_height * output_width +
                       (out_h + i) * output_width + (out_w + j),
                       avg_val[i, j])

def triton_avg_pool2d(input_tensor, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w):
    """
    Triton implementation of 2D Average Pooling
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Get dimensions
    batch_size, channels, input_height, input_width = input_tensor.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding_h - kernel_h) // stride_h + 1
    output_width = (input_width + 2 * padding_w - kernel_w) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    CHANNELS_PER_BLOCK = 1
    
    # Grid dimensions
    grid = (
        batch_size,
        channels,
        (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
    # Launch kernel
    avg_pool2d_kernel_optimized[grid](
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
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
        CHANNELS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernels for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer with custom Triton implementation.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool2d(
            x,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding
        )