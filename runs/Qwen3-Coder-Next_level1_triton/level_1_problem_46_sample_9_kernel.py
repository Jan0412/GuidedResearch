import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avgpool3d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # B
    channels,  # C
    in_d, in_h, in_w,  # Input dimensions
    out_d, out_h, out_w,  # Output dimensions
    kernel_d, kernel_h, kernel_w,  # Kernel dimensions
    stride_d, stride_h, stride_w,  # Stride dimensions
    pad_d, pad_h, pad_w,  # Padding dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output tensor indices
    out_idx = tl.program_id(0)
    
    # Decode the linear output index into (b, c, od, oh, ow)
    ow = out_idx % out_w
    out_idx = out_idx // out_w
    oh = out_idx % out_h
    out_idx = out_idx // out_h
    od = out_idx % out_d
    out_idx = out_idx // out_d
    c = out_idx % channels
    b = out_idx // channels
    
    # Calculate the starting position in the input tensor
    start_d = od * stride_d - pad_d
    start_h = oh * stride_h - pad_h
    start_w = ow * stride_w - pad_w
    
    # Compute the average
    sum_val = 0.0
    count = 0
    
    # Iterate over the kernel window
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position
                d = start_d + kd
                h = start_h + kh
                w = start_w + kw
                
                # Check bounds (only include valid positions)
                if (d >= 0 and d < in_d and 
                    h >= 0 and h < in_h and 
                    w >= 0 and w < in_w):
                    # Calculate input pointer offset
                    offset = (((b * channels + c) * in_d + d) * in_h + h) * in_w + w
                    val = tl.load(x_ptr + offset)
                    sum_val += val
                    count += 1
    
    # Compute average (avoid division by zero)
    avg = sum_val / count if count > 0 else 0.0
    
    # Store result
    tl.store(out_ptr + out_idx, avg)


def triton_avgpool3d(x, kernel_size, stride, padding):
    """
    Applies 3D average pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, depth, height, width)
        kernel_size: int or tuple of ints
        stride: int or tuple of ints
        padding: int or tuple of ints
        
    Returns:
        Output tensor after 3D average pooling
    """
    # Handle tuple inputs
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    
    # Extract dimensions
    batch_size, channels, in_d, in_h, in_w = x.shape
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    
    # Calculate output dimensions
    out_d = (in_d + 2 * pad_d - kernel_d) // stride_d + 1
    out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, channels, out_d, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Flatten the output for 1D grid
    total_elements = batch_size * channels * out_d * out_h * out_w
    
    # Configure kernel launch
    BLOCK_SIZE = 128
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    avgpool3d_kernel[grid](
        x, out,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using Triton kernel.
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
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avgpool3d(x, self.kernel_size, self.stride, self.padding)