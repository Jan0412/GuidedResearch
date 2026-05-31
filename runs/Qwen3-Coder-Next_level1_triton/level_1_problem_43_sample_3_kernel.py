import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    channels,  # Number of channels
    in_d, in_h, in_w,  # Input dimensions
    out_d, out_h, out_w,  # Output dimensions
    kernel_d, kernel_h, kernel_w,  # Kernel size
    stride_d, stride_h, stride_w,  # Stride
    pad_d, pad_h, pad_w,  # Padding
    dil_d, dil_h, dil_w,  # Dilation
    n_elements,  # Total number of output elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output index
    output_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < n_elements
    
    # Decode output indices to (b, c, od, oh, ow)
    ow = output_idx % out_w
    tmp = output_idx // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    od = tmp % out_d
    tmp = tmp // out_d
    c = tmp % channels
    b = tmp // channels
    
    # Compute input starting position for this output
    in_d_start = od * stride_d - pad_d
    in_h_start = oh * stride_h - pad_h
    in_w_start = ow * stride_w - pad_w
    
    # Initialize max value
    max_val = -float('inf')
    
    # Iterate over kernel window
    for kd in range(kernel_d):
        in_d_pos = in_d_start + kd * dil_d
        for kh in range(kernel_h):
            in_h_pos = in_h_start + kh * dil_h
            for kw in range(kernel_w):
                in_w_pos = in_w_start + kw * dil_w
                
                # Check if within input bounds
                valid = (in_d_pos >= 0) & (in_d_pos < in_d) & \
                        (in_h_pos >= 0) & (in_h_pos < in_h) & \
                        (in_w_pos >= 0) & (in_w_pos < in_w)
                
                if valid:
                    # Compute input index
                    input_idx = ((b * channels + c) * in_d + in_d_pos) * in_h * in_w + \
                               in_h_pos * in_w + in_w_pos
                    # Load and update max
                    val = tl.load(x_ptr + input_idx)
                    max_val = tl.maximum(max_val, val)
    
    # Store result
    tl.store(out_ptr + output_idx, max_val, mask=mask)


def triton_maxpool3d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
    ceil_mode: bool = False
) -> torch.Tensor:
    """
    Applies 3D max pooling using Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Extract dimensions
    batch_size, channels, in_d, in_h, in_w = x.shape
    
    # Set stride to kernel_size if not specified
    if stride is None:
        stride = kernel_size
    
    # Calculate output dimensions
    if ceil_mode:
        out_d = int((in_d + 2 * padding - dilation * (kernel_size - 1) + stride - 1) // stride + 1)
        out_h = int((in_h + 2 * padding - dilation * (kernel_size - 1) + stride - 1) // stride + 1)
        out_w = int((in_w + 2 * padding - dilation * (kernel_size - 1) + stride - 1) // stride + 1)
    else:
        out_d = int((in_d + 2 * padding - dilation * (kernel_size - 1)) // stride + 1)
        out_h = int((in_h + 2 * padding - dilation * (kernel_size - 1)) // stride + 1)
        out_w = int((in_w + 2 * padding - dilation * (kernel_size - 1)) // stride + 1)
    
    # Create output tensor
    out = torch.empty(batch_size, channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Total output elements
    n_elements = out.numel()
    BLOCK_SIZE = 256
    
    # Grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    maxpool3d_kernel[grid](
        x, out,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, kernel_size, kernel_size,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model with MaxPool3d replaced by custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, 
                 dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for reference
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
        # Only support basic maxpool3d without return_indices for Triton implementation
        assert not return_indices, "Triton implementation does not support return_indices=True"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D using custom Triton kernel.
        """
        return triton_maxpool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode
        )