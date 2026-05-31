import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size, channels, 
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output indices
    out_idx = tl.program_id(0)
    
    # Calculate which batch, channel, and spatial position this thread processes
    # Layout: [batch, channel, out_d, out_h, out_w]
    total_out = batch_size * channels * out_d * out_h * out_w
    
    if out_idx >= total_out:
        return
        
    # Decode output indices
    tmp = out_idx
    out_w_idx = tmp % out_w
    tmp //= out_w
    out_h_idx = tmp % out_h
    tmp //= out_h
    out_d_idx = tmp % out_d
    tmp //= out_d
    ch_idx = tmp % channels
    batch_idx = tmp // channels
    
    # Calculate input starting position (top-left corner of kernel)
    in_d_start = out_d_idx * stride_d - pad_d
    in_h_start = out_h_idx * stride_h - pad_h
    in_w_start = out_w_idx * stride_w - pad_w
    
    # Calculate input dimension offsets
    input_stride_d = in_h * in_w
    input_stride_h = in_w
    input_stride_w = 1
    
    # Calculate batch and channel offset
    batch_offset = batch_idx * channels * in_d * in_h * in_w
    channel_offset = ch_idx * in_d * in_h * in_w
    
    max_val = -float('inf')
    
    # Iterate over kernel window
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate actual input coordinates
                in_d_pos = in_d_start + kd * dil_d
                in_h_pos = in_h_start + kh * dil_h
                in_w_pos = in_w_start + kw * dil_w
                
                # Check if within bounds
                if (0 <= in_d_pos < in_d and 
                    0 <= in_h_pos < in_h and 
                    0 <= in_w_pos < in_w):
                    # Calculate input index
                    input_idx = (batch_offset + channel_offset + 
                                in_d_pos * input_stride_d + 
                                in_h_pos * input_stride_h + 
                                in_w_pos * input_stride_w)
                    val = tl.load(x_ptr + input_idx)
                    max_val = tl.maximum(max_val, val)
    
    # Store result
    tl.store(out_ptr + out_idx, max_val)


def triton_maxpool3d(x, kernel_size, stride, padding, dilation, ceil_mode):
    """
    Triton implementation of MaxPool3d.
    """
    batch_size, channels, in_d, in_h, in_w = x.shape
    
    # Convert to tuples if needed
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation
    
    # Calculate output dimensions
    if ceil_mode:
        out_d = int((in_d - (kernel_d - 1) * dil_d + 2 * pad_d + stride_d - 1) // stride_d + 1)
        out_h = int((in_h - (kernel_h - 1) * dil_h + 2 * pad_h + stride_h - 1) // stride_h + 1)
        out_w = int((in_w - (kernel_w - 1) * dil_w + 2 * pad_w + stride_w - 1) // stride_w + 1)
    else:
        out_d = (in_d + 2 * pad_d - (kernel_d - 1) * dil_d - 1) // stride_d + 1
        out_h = (in_h + 2 * pad_h - (kernel_h - 1) * dil_h - 1) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (kernel_w - 1) * dil_w - 1) // stride_w + 1
    
    # Create output tensor
    out = torch.empty(batch_size, channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Calculate grid size
    total_out = batch_size * channels * out_d * out_h * out_w
    BLOCK_SIZE = 256
    grid = ((total_out + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    maxpool3d_kernel[grid](
        x, out,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for MaxPool3d.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, 
                 dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the optimized Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices. Not supported in Triton version.
            ceil_mode (bool, optional): When True, use ceil for output size calculation. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using Triton kernel.
        """
        if self.return_indices:
            # Return original output and placeholder for indices
            output = triton_maxpool3d(x, self.kernel_size, self.stride, 
                                     self.padding, self.dilation, self.ceil_mode)
            return output, torch.empty_like(output, dtype=torch.int64)
        else:
            return triton_maxpool3d(x, self.kernel_size, self.stride, 
                                   self.padding, self.dilation, self.ceil_mode)