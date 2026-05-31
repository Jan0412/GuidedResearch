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
    BLOCK_SIZE: tl.constexpr
):
    # Get thread index
    idx = tl.program_id(0)
    
    # Calculate total output elements
    total_output_elements = batch_size * channels * output_depth * output_height * output_width
    
    if idx >= total_output_elements:
        return
        
    # Convert linear index to multi-dimensional indices
    temp = idx
    out_width_idx = temp % output_width
    temp //= output_width
    out_height_idx = temp % output_height
    temp //= output_height
    out_depth_idx = temp % output_depth
    temp //= output_depth
    channel_idx = temp % channels
    batch_idx = temp // channels
    
    # Calculate input region boundaries
    d_start = out_depth_idx * stride - padding
    h_start = out_height_idx * stride - padding
    w_start = out_width_idx * stride - padding
    
    # Initialize sum
    sum_val = tl.zeros([1], dtype=tl.float32)
    count = 0
    
    # Iterate through kernel
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                d = d_start + kd
                h = h_start + kh
                w = w_start + kw
                
                # Check bounds
                if d >= 0 and d < input_depth and h >= 0 and h < input_height and w >= 0 and w < input_width:
                    # Calculate input index
                    input_idx = (
                        batch_idx * (channels * input_depth * input_height * input_width) +
                        channel_idx * (input_depth * input_height * input_width) +
                        d * (input_height * input_width) +
                        h * input_width +
                        w
                    )
                    
                    # Load value and accumulate
                    val = tl.load(input_ptr + input_idx, mask=True)
                    sum_val += val
                    count += 1
    
    # Calculate average
    avg_val = sum_val / count if count > 0 else 0.0
    
    # Store result
    output_idx = idx
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
        # Ensure input is on GPU
        if not x.is_cuda:
            x = x.cuda()
            
        # Calculate output dimensions
        batch_size, channels, input_depth, input_height, input_width = x.shape
        
        output_depth = (input_depth + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(
            batch_size, channels, output_depth, output_height, output_width,
            dtype=torch.float32, device=x.device
        )
        
        # Prepare input tensor for contiguous memory access
        x_contiguous = x.contiguous()
        
        # Calculate total elements in output
        total_output_elements = batch_size * channels * output_depth * output_height * output_width
        
        if total_output_elements == 0:
            return output
            
        # Define block size
        BLOCK_SIZE = 128
        
        # Calculate grid size
        grid_size = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        avgpool3d_kernel[grid_size](
            x_contiguous,
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