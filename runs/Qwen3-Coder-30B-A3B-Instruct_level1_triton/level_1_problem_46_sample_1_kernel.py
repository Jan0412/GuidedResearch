import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avgpool3d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Calculate global indices
    batch_idx = block_idx // (channels * output_depth * output_height * output_width)
    remaining = block_idx % (channels * output_depth * output_height * output_width)
    channel_idx = remaining // (output_depth * output_height * output_width)
    remaining = remaining % (output_depth * output_height * output_width)
    depth_idx = remaining // (output_height * output_width)
    remaining = remaining % (output_height * output_width)
    height_idx = remaining // output_width
    width_idx = remaining % output_width
    
    # Check bounds
    if batch_idx >= batch_size or channel_idx >= channels or depth_idx >= output_depth or height_idx >= output_height or width_idx >= output_width:
        return
        
    # Calculate input start positions
    input_depth_start = depth_idx * stride - padding
    input_height_start = height_idx * stride - padding
    input_width_start = width_idx * stride - padding
    
    # Initialize accumulator
    sum_val = tl.zeros((1,), dtype=tl.float32)
    count = 0
    
    # Iterate through kernel
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                input_d = input_depth_start + kd
                input_h = input_height_start + kh
                input_w = input_width_start + kw
                
                # Check if within bounds
                if (input_d >= 0 and input_d < input_depth and 
                    input_h >= 0 and input_h < input_height and 
                    input_w >= 0 and input_w < input_width):
                    
                    input_offset = (batch_idx * channels * input_depth * input_height * input_width +
                                   channel_idx * input_depth * input_height * input_width +
                                   input_d * input_height * input_width +
                                   input_h * input_width +
                                   input_w)
                    
                    sum_val += tl.load(input_ptr + input_offset, mask=True)
                    count += 1
    
    # Calculate average
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
        
    # Store result
    output_offset = (batch_idx * channels * output_depth * output_height * output_width +
                     channel_idx * output_depth * output_height * output_width +
                     depth_idx * output_height * output_width +
                     height_idx * output_width +
                     width_idx)
    
    tl.store(output_ptr + output_offset, avg_val)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        batch_size, channels, input_depth, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_depth = (input_depth + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Calculate total elements in output
        total_elements = batch_size * channels * output_depth * output_height * output_width
        
        # Set up block size
        BLOCK_SIZE = 1024
        
        # Determine grid size
        grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        avgpool3d_kernel[grid_size](
            x,
            output,
            batch_size,
            channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            self.kernel_size,
            self.stride,
            self.padding,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output