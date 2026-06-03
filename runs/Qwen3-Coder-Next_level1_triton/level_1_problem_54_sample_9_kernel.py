import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias tensor pointer (optional)
    out_ptr,  # Output tensor pointer
    batch_size,  # B
    in_channels,  # C_in
    out_channels,  # C_out
    depth,  # D
    height,  # H
    width,  # W
    kernel_depth,  # Kd
    kernel_height,  # Kh
    kernel_width,  # Kw
    stride_d,  # Sd
    stride_h,  # Sh
    stride_w,  # Sw
    pad_d,  # Pd
    pad_h,  # Ph
    pad_w,  # Pw
    dilation_d,  # Dd
    dilation_h,  # Dh
    dilation_w,  # Dw
    n_elements,  # Total output elements
    BLOCK_SIZE: tl.constexpr,
):
    # Get output position
    out_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_idx < n_elements
    
    # Decode output index into (b, oc, od, oh, ow)
    ow = out_idx % width
    tmp = out_idx // width
    oh = tmp % height
    tmp = tmp // height
    od = tmp % depth
    tmp = tmp // depth
    oc = tmp % out_channels
    b = tmp // out_channels
    
    # Compute the starting position in the input
    id_start = od * stride_d - pad_d
    ih_start = oh * stride_h - pad_h
    iw_start = ow * stride_w - pad_w
    
    # Accumulator for the convolution
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(in_channels):
        # Loop over kernel dimensions
        for kd in range(kernel_depth):
            id = id_start + kd * dilation_d
            valid_d = (id >= 0) & (id < depth)
            
            for kh in range(kernel_height):
                ih = ih_start + kh * dilation_h
                valid_h = (ih >= 0) & (ih < height)
                
                for kw in range(kernel_width):
                    iw = iw_start + kw * dilation_w
                    valid_w = (iw >= 0) & (iw < width)
                    
                    # Combined validity mask
                    valid = valid_d & valid_h & valid_w
                    
                    # Compute input index
                    in_idx = b * (in_channels * depth * height * width) + \
                             ic * (depth * height * width) + \
                             id * (height * width) + \
                             ih * width + \
                             iw
                    
                    # Compute weight index
                    w_idx = oc * (in_channels * kernel_depth * kernel_height * kernel_width) + \
                            ic * (kernel_depth * kernel_height * kernel_width) + \
                            kd * (kernel_height * kernel_width) + \
                            kh * kernel_width + \
                            kw
                    
                    # Load input and weight values (with zero for out of bounds)
                    x_val = tl.load(x_ptr + in_idx, mask=valid, other=0.0)
                    w_val = tl.load(w_ptr + w_idx)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + oc)
        acc += bias
    
    # Store result
    tl.store(out_ptr + out_idx, acc, mask=mask)


def triton_conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 3D convolution.
    Currently supports groups=1 (standard convolution).
    """
    # Extract shapes
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Validate parameters
    assert groups == 1, "Group convolution not supported in this Triton implementation"
    
    # Compute output dimensions
    stride_d = stride_h = stride_w = stride
    pad_d = pad_h = pad_w = padding
    dilation_d = dilation_h = dilation_w = dilation
    
    out_d = (depth + 2 * pad_d - dilation_d * (kernel_depth - 1) - 1) // stride_d + 1
    out_h = (height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out_shape = (batch_size, out_channels, out_d, out_h, out_w)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Flatten output for 1D kernel launch
    n_elements = out.numel()
    BLOCK_SIZE = 128
    
    # Grid configuration
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_depth, kernel_height, kernel_width,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel.
    Uses optimized Triton kernel for computation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        return triton_conv3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )