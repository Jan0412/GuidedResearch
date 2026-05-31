import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    # Input tensor dimensions
    batch_size, channels, 
    in_d, in_h, in_w,
    # Output tensor dimensions
    out_d, out_h, out_w,
    # Pooling parameters
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dilation_d, dilation_h, dilation_w,
    # Block sizes for parallelization
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    # Calculate output indices
    batch_idx = tl.program_id(0)
    out_d_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    channel_block = tl.program_id(4)
    
    # Calculate input starting position (top-left corner of the pooling window)
    input_d_start = out_d_idx * stride_d - pad_d
    input_h_start = out_h_idx * stride_h - pad_h
    input_w_start = out_w_idx * stride_w - pad_w
    
    # Compute the range of channels to process in this block
    channel_start = channel_block * BLOCK_CHANNELS
    channel_mask = (tl.arange(0, BLOCK_CHANNELS) < channels) & (channel_start + tl.arange(0, BLOCK_CHANNELS) >= 0) & (channel_start + tl.arange(0, BLOCK_CHANNELS) < channels)
    
    # Initialize output with -inf for max operation
    output = tl.full([BLOCK_CHANNELS], -float('inf'), dtype=tl.float32)
    
    # Iterate through the pooling window
    for kd in range(kernel_d):
        input_d = input_d_start + kd * dilation_d
        d_in_bounds = (input_d >= 0) & (input_d < in_d)
        
        for kh in range(kernel_h):
            input_h = input_h_start + kh * dilation_h
            h_in_bounds = (input_h >= 0) & (input_h < in_h)
            
            for kw in range(kernel_w):
                input_w = input_w_start + kw * dilation_w
                w_in_bounds = (input_w >= 0) & (input_w < in_w)
                
                # Check if this position is within input bounds
                in_bounds = d_in_bounds & h_in_bounds & w_in_bounds
                
                if tl.static_cast(tl.int1, in_bounds):
                    # Calculate input pointer offset
                    offset = (
                        batch_idx * channels * in_d * in_h * in_w +
                        channel_start * in_d * in_h * in_w +
                        input_d * in_h * in_w +
                        input_h * in_w +
                        input_w
                    )
                    
                    # Load data
                    x_vals = tl.load(
                        x_ptr + offset,
                        mask=channel_mask,
                        other=-float('inf')
                    )
                    
                    # Update max
                    output = tl.maximum(output, x_vals)
    
    # Store output
    out_offset = (
        batch_idx * channels * out_d * out_h * out_w +
        channel_start * out_d * out_h * out_w +
        out_d_idx * out_h * out_w +
        out_h_idx * out_w +
        out_w_idx
    )
    
    tl.store(
        out_ptr + out_offset,
        output,
        mask=channel_mask
    )


def triton_maxpool3d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
    return_indices: bool = False,
    ceil_mode: bool = False
):
    """
    Triton implementation of 3D max pooling.
    """
    # Ensure input is contiguous and on CUDA
    x = x.contiguous()
    assert x.is_cuda, "Input tensor must be on CUDA device."
    
    # Get input dimensions
    batch_size, channels, in_d, in_h, in_w = x.shape
    
    # Set stride to kernel_size if not specified
    if stride is None:
        stride = kernel_size
    
    # Calculate output dimensions
    # Using the standard formula: (input + 2*padding - dilation*(kernel-1) - 1) // stride + 1
    # But also support ceil_mode
    def calc_out_dim(in_dim, k, s, p, d, ceil_mode):
        if ceil_mode:
            return (in_dim + 2 * p - d * (k - 1) - 1 + s - 1) // s + 1
        else:
            return (in_dim + 2 * p - d * (k - 1) - 1) // s + 1
    
    out_d = calc_out_dim(in_d, kernel_size, stride, padding, dilation, ceil_mode)
    out_h = calc_out_dim(in_h, kernel_size, stride, padding, dilation, ceil_mode)
    out_w = calc_out_dim(in_w, kernel_size, stride, padding, dilation, ceil_mode)
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Handle 1D case where dimensions might be 1
    kernel_d = kernel_h = kernel_w = kernel_size
    stride_d = stride_h = stride_w = stride
    pad_d = pad_h = pad_w = padding
    dilation_d = dilation_h = dilation_w = dilation
    
    # Set block sizes for efficient parallelization
    BLOCK_SIZE_D = 1
    BLOCK_SIZE_H = 1
    BLOCK_SIZE_W = 1
    BLOCK_CHANNELS = min(32, channels)  # Process multiple channels per block
    
    # Grid dimensions: [batch, out_d, out_h, out_w, channels//BLOCK_CHANNELS]
    grid = (
        batch_size,
        out_d,
        out_h,
        out_w,
        (channels + BLOCK_CHANNELS - 1) // BLOCK_CHANNELS
    )
    
    # Launch kernel
    maxpool3d_kernel[grid](
        x, out,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_CHANNELS=BLOCK_CHANNELS
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Max Pooling 3D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D using Triton kernel.
        """
        return triton_maxpool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=self.return_indices,
            ceil_mode=self.ceil_mode
        )