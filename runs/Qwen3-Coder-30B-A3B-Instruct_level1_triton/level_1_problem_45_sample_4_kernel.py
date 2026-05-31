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
):
    # Get the batch and channel index for this program
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting position for this thread block
    output_h_start = tl.program_id(2) * BLOCK_SIZE
    output_w_start = tl.program_id(3) * BLOCK_SIZE
    
    # Shared memory for reduction
    shared_mem = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Loop over the output dimensions
    for out_h in range(output_h_start, min(output_h_start + BLOCK_SIZE, output_height)):
        for out_w in range(output_w_start, min(output_w_start + BLOCK_SIZE, output_width)):
            # Initialize accumulator
            acc = tl.zeros((1,), dtype=tl.float32)
            count = 0
            
            # Loop over kernel
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Calculate input position
                    ih = out_h * stride_h - padding_h + kh
                    iw = out_w * stride_w - padding_w + kw
                    
                    # Check bounds
                    if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                        # Load value
                        val = tl.load(input_ptr + 
                                    batch_idx * (channels * input_height * input_width) +
                                    channel_idx * (input_height * input_width) +
                                    ih * input_width + iw,
                                    mask=True, other=0.0)
                        acc += val
                        count += 1
            
            # Compute average
            if count > 0:
                avg_val = acc / count
            else:
                avg_val = 0.0
                
            # Store result
            tl.store(output_ptr + 
                    batch_idx * (channels * output_height * output_width) +
                    channel_idx * (output_height * output_width) +
                    out_h * output_width + out_w,
                    avg_val)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

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
        batch_size, channels, height, width = x.shape
        
        # Calculate output dimensions
        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Define block size for optimization
        BLOCK_SIZE = 16
        
        # Grid configuration
        grid = (
            batch_size,
            channels,
            (output_height + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (output_width + BLOCK_SIZE - 1) // BLOCK_SIZE
        )
        
        # Launch kernel
        avg_pool2d_kernel[grid](
            x,
            output,
            batch_size,
            channels,
            height,
            width,
            output_height,
            output_width,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output