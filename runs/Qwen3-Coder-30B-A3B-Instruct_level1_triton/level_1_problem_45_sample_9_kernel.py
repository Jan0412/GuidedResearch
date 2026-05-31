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
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    
    # Calculate output dimensions
    # Each block handles one channel per batch
    if channel_idx >= channels:
        return
    
    # Shared memory for reduction
    shared_data = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Loop over output width
    for output_x in range(tl.cdiv(output_width, BLOCK_SIZE)):
        # Calculate starting position
        start_x = output_x * BLOCK_SIZE * stride_w
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        count = 0
        
        # Process kernel elements
        for ky in range(kernel_h):
            for kx in range(kernel_w):
                # Calculate input coordinates
                input_y = output_y * stride_h + ky - padding_h
                input_x = start_x + kx - padding_w
                
                # Check bounds
                if (input_y >= 0 and input_y < input_height and 
                    input_x >= 0 and input_x < input_width):
                    
                    # Load from global memory
                    input_offset = (batch_idx * channels * input_height * input_width + 
                                  channel_idx * input_height * input_width + 
                                  input_y * input_width + input_x)
                    
                    # Load with bounds checking
                    if (output_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < output_width):
                        load_indices = input_y * input_width + (start_x + tl.arange(0, BLOCK_SIZE) + kx - padding_w)
                        # Ensure we don't go out of bounds for the input
                        valid_mask = (load_indices >= 0) & (load_indices < input_width)
                        if valid_mask.any():
                            acc += tl.load(input_ptr + input_offset + load_indices, mask=valid_mask, other=0.0)
                            count += tl.sum(valid_mask.to(tl.int32))
        
        # Store results
        output_offset = (batch_idx * channels * output_height * output_width + 
                        channel_idx * output_height * output_width + 
                        output_y * output_width + output_x * BLOCK_SIZE)
        
        # Store accumulated values
        if output_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < output_width:
            store_indices = output_y * output_width + (output_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
            valid_store_mask = store_indices < output_height * output_width
            # Compute average
            avg_val = acc / count if count > 0 else 0.0
            tl.store(output_ptr + output_offset, avg_val, mask=valid_store_mask)

# More efficient version using proper tiling approach
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
    TILE_SIZE: tl.constexpr
):
    # Thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    tile_y = tl.program_id(2)
    tile_x = tl.program_id(3)
    
    # Ensure we're within bounds
    if tile_y >= output_height or tile_x >= output_width:
        return
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    count = 0
    
    # Process kernel window
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            # Calculate input coordinates
            input_y = tile_y * stride_h + ky - padding_h
            input_x = tile_x * stride_w + kx - padding_w
            
            # Check if this position is valid
            if (input_y >= 0 and input_y < input_height and 
                input_x >= 0 and input_x < input_width):
                
                # Calculate input offset
                input_offset = (batch_idx * channels * input_height * input_width + 
                              channel_idx * input_height * input_width + 
                              input_y * input_width + input_x)
                
                # Load value and accumulate
                val = tl.load(input_ptr + input_offset, mask=True)
                acc += val
                count += 1
    
    # Compute average
    avg_val = acc / count if count > 0 else 0.0
    
    # Store result
    output_offset = (batch_idx * channels * output_height * output_width + 
                    channel_idx * output_height * output_width + 
                    tile_y * output_width + tile_x)
    
    tl.store(output_ptr + output_offset, avg_val)

def triton_avg_pool2d(input_tensor, kernel_size, stride=None, padding=0):
    """
    Triton implementation of 2D Average Pooling
    """
    if stride is None:
        stride = kernel_size
    
    batch_size, channels, input_height, input_width = input_tensor.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Define grid dimensions
    grid = (
        batch_size,       # Batch dimension
        channels,         # Channel dimension  
        output_height,    # Output height tiles
        output_width      # Output width tiles
    )
    
    # Launch kernel
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    TILE_SIZE = 16
    
    avg_pool2d_kernel_optimized[grid](
        input_tensor,
        output,
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        kernel_size,
        stride,
        stride,
        padding,
        padding,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
        TILE_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer with Triton optimization.

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
        Applies 2D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)