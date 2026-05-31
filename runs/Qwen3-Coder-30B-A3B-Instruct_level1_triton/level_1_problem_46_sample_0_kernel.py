import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

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
    # Get thread index
    idx = tl.program_id(0)
    
    # Calculate which output element this thread handles
    if idx >= batch_size * channels * output_depth * output_height * output_width:
        return
        
    # Decompose the linear index into multi-dimensional indices
    ow = idx % output_width
    oh = (idx // output_width) % output_height
    od = (idx // (output_width * output_height)) % output_depth
    oc = (idx // (output_width * output_height * output_depth)) % channels
    ob = (idx // (output_width * output_height * output_depth * channels)) % batch_size
    
    # Calculate input region boundaries
    input_start_d = od * stride - padding
    input_start_h = oh * stride - padding
    input_start_w = ow * stride - padding
    
    # Initialize accumulator
    sum_val = tl.zeros([1], dtype=tl.float32)
    count = 0
    
    # Iterate through kernel
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                input_d = input_start_d + kd
                input_h = input_start_h + kh
                input_w = input_start_w + kw
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and 
                    input_h >= 0 and input_h < input_height and 
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input index
                    input_idx = (ob * (channels * input_depth * input_height * input_width) +
                                oc * (input_depth * input_height * input_width) +
                                input_d * (input_height * input_width) +
                                input_h * input_width +
                                input_w)
                    
                    sum_val += tl.load(input_ptr + input_idx, mask=True)
                    count += 1
    
    # Compute average
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Store result
    output_idx = (ob * (channels * output_depth * output_height * output_width) +
                 oc * (output_depth * output_height * output_width) +
                 od * (output_height * output_width) +
                 oh * output_width +
                 ow)
    
    tl.store(output_ptr + output_idx, avg_val)

class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels for 3D Average Pooling.
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
        batch_size, channels, depth, height, width = x.shape
        
        # Calculate output dimensions
        output_depth = (depth + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_depth, output_height, output_width, 
                           dtype=torch.float32, device=x.device)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Calculate total elements in output
        total_elements = batch_size * channels * output_depth * output_height * output_width
        
        if total_elements == 0:
            return output
            
        # Define block size
        BLOCK_SIZE = 1024
        
        # Calculate grid size
        grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        avgpool3d_kernel[grid_size](
            x,
            output,
            batch_size,
            channels,
            depth,
            height,
            width,
            output_depth,
            output_height,
            output_width,
            self.kernel_size,
            self.stride,
            self.padding,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output