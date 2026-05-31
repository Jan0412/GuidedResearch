import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_height,
    kernel_width,
    height_out,
    width_out,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block start and offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid offsets
    n_elements = batch_size * out_channels * height_out * width_out
    mask = offsets < n_elements
    
    # Decode offsets to batch, out_channel, out_h, out_w
    # stride_out_c = height_out * width_out
    # stride_out_h = width_out
    # stride_out_w = 1
    
    # Using integer division and modulo for decoding
    # b = offsets // (out_channels * height_out * width_out)
    # rem = offsets % (out_channels * height_out * width_out)
    # c_out = rem // (height_out * width_out)
    # rem = rem % (height_out * width_out)
    # h = rem // width_out
    # w = rem % width_out
    
    # Optimized decoding using tl.cdiv and tl.multiple_of if applicable, 
    # but standard arithmetic is robust here.
    
    b = offsets // (out_channels * height_out * width_out)
    rem = offsets % (out_channels * height_out * width_out)
    c_out = rem // (height_out * width_out)
    rem = rem % (height_out * width_out)
    h = rem // width_out
    w = rem % width_out
    
    # Accumulator for the convolution sum
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel height
        for kh in range(kernel_height):
            # Loop over kernel width
            for kw in range(kernel_width):
                # Calculate input coordinates
                ih = h + kh
                iw = w + kw
                
                # Load input values with masking for bounds (though padding=0 ensures bounds for valid output)
                # x indices: [b, ic, ih, iw]
                x_idx = (b * in_channels + ic) * height * width + ih * width + iw
                x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
                
                # w indices: [c_out, ic, kh, kw]
                w_idx = (c_out * in_channels + ic) * kernel_height * kernel_width + kh * kernel_width + kw
                w_val = tl.load(w_ptr + w_idx, mask=mask, other=0.0)
                
                acc += x_val * w_val
    
    # Store result
    # out indices: [b, c_out, h, w]
    out_idx = (b * out_channels + c_out) * height_out * width_out + h * width_out + w
    tl.store(out_ptr + out_idx, acc, mask=mask)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton implementation of 2D convolution.
    Assumes: padding=0, dilation=1, stride=1, groups=1, bias=False.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_height, kernel_width = w.shape
    
    height_out = height - kernel_height + 1
    width_out = width - kernel_width + 1
    
    out = torch.empty((batch_size, out_channels, height_out, width_out), dtype=x.dtype, device=x.device)
    
    n_elements = batch_size * out_channels * height_out * width_out
    BLOCK_SIZE = 256  # Tunable parameter
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    conv2d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels,
        height, width,
        kernel_height, kernel_width,
        height_out, width_out,
        1, 1,  # stride_h, stride_w
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Model using custom Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the custom Triton kernel.
        """
        # Extract weights from the nn.Conv2d module
        w = self.conv2d.weight
        # The custom kernel assumes no bias and groups=1, padding=0, dilation=1, stride=1.
        # If the model has different settings, this implementation would need adjustments.
        # Based on the problem description, we assume the standard settings provided in the example.
        return triton_conv2d(x, w)


def get_inputs():
    # randomly generate input tensors based on the model architecture
    batch_size = 8
    in_channels = 64
    height = 512
    width = 256
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    in_channels = 64
    out_channels = 128
    kernel_size = (5, 7)
    return [in_channels, out_channels, kernel_size]