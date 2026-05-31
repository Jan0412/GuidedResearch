import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the batch and channel indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    
    # Calculate output dimensions
    output_size = output_height * output_width
    
    # Shared memory for input tile
    tile_size = BLOCK_SIZE
    tile_h = tile_size
    tile_w = tile_size
    
    # Grid for output position
    output_pos = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    if bias_ptr != 0:
        acc += tl.load(bias_ptr + out_ch_idx, mask=out_ch_idx < out_channels, other=0.0)
    
    # Loop over input channels
    for ic in range(0, in_channels, GROUP_SIZE_M):
        # Load weights for this channel
        w_offset = ic * out_channels * kernel_h * kernel_w + out_ch_idx * kernel_h * kernel_w
        w_ptr = weight_ptr + w_offset
        
        # Load input tile
        input_offset = batch_idx * in_channels * input_height * input_width + ic * input_height * input_width
        input_ptr_local = input_ptr + input_offset
        
        # For each output position, compute convolution
        for oh in range(output_height):
            for ow in range(output_width):
                # Compute input coordinates
                ih = oh * stride_h - pad_h
                iw = ow * stride_w - pad_w
                
                # Accumulate over kernel
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        # Apply dilation
                        ih_k = ih + kh * dilation_h
                        iw_k = iw + kw * dilation_w
                        
                        # Check bounds
                        if ih_k >= 0 and ih_k < input_height and iw_k >= 0 and iw_k < input_width:
                            # Compute input index
                            input_idx = ih_k * input_width + iw_k
                            input_val = tl.load(input_ptr_local + input_idx, mask=True, other=0.0)
                            
                            # Compute weight index
                            w_idx = kh * kernel_w + kw
                            w_val = tl.load(w_ptr + w_idx, mask=True, other=0.0)
                            
                            # Accumulate
                            acc += input_val * w_val
    
    # Store output
    output_offset = batch_idx * out_channels * output_height * output_width + out_ch_idx * output_height * output_width
    output_ptr_local = output_ptr + output_offset
    
    # Store results
    for i in range(BLOCK_SIZE):
        if output_pos[i] < output_size:
            pos = output_pos[i]
            oh = pos // output_width
            ow = pos % output_width
            out_idx = oh * output_width + ow
            tl.store(output_ptr_local + out_idx, acc[i], mask=output_pos[i] < output_size)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of ConvTranspose2d
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    
    output_height = (input_height - 1) * stride_h - 2 * pad_h + (kernel_h - 1) * dilation_h + 1
    output_width = (input_width - 1) * stride_w - 2 * pad_w + (kernel_w - 1) * dilation_w + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block size and group size
    BLOCK_SIZE = 128
    GROUP_SIZE_M = 32
    
    # Launch kernel
    grid = (
        batch_size,           # Batch dimension
        out_channels,         # Output channels
        math.ceil(output_height * output_width / BLOCK_SIZE)  # Output positions
    )
    
    # Kernel launch
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and square kernel, supporting dilation, padding, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square, e.g., 3 for a 3x3 kernel).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in). 

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use the Triton kernel implementation
        return triton_conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation)
        )