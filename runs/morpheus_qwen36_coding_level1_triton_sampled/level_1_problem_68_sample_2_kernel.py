import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    depth, width, height,
    kernel_depth, kernel_width, kernel_height,
    stride_d, stride_w, stride_h,
    padding_d, padding_w, padding_h,
    output_padding_d, output_padding_w, output_padding_h,
    groups,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate output dimensions
    depth_out = (depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d
    width_out = (width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w
    height_out = (height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h
    
    # Total number of output elements
    n_elements = batch_size * out_channels * depth_out * width_out * height_out
    
    # Grid over output elements
    idx = tl.program_id(0)
    if idx >= n_elements:
        return
    
    # Calculate output coordinates
    b = idx // (out_channels * depth_out * width_out * height_out)
    rest = idx % (out_channels * depth_out * width_out * height_out)
    c_out = rest // (depth_out * width_out * height_out)
    rest = rest % (depth_out * width_out * height_out)
    d_out = rest // (width_out * height_out)
    rest = rest % (width_out * height_out)
    w_out = rest // height_out
    h_out = rest % height_out
    
    # Calculate base input coordinates
    d_in_base = d_out // stride_d - padding_d
    w_in_base = w_out // stride_w - padding_w
    h_in_base = h_out // stride_h - padding_h
    
    # Group index
    group = c_out // (out_channels // groups)
    c_in_base = group * (in_channels // groups)
    
    acc = 0.0
    
    # Iterate over kernel dimensions
    for kd in range(kernel_depth):
        for kw in range(kernel_width):
            for kh in range(kernel_height):
                d_in = d_in_base + kd
                w_in = w_in_base + kw
                h_in = h_in_base + kh
                
                # Check bounds
                if d_in >= 0 and d_in < depth and w_in >= 0 and w_in < width and h_in >= 0 and h_in < height:
                    # Iterate over input channels in the group
                    for ci in range(in_channels // groups):
                        c_in = c_in_base + ci
                        
                        # Load input value
                        x_idx = ((b * in_channels + c_in) * depth + d_in) * width * height + w_in * height + h_in
                        x = tl.load(x_ptr + x_idx)
                        
                        # Load weight value
                        w_idx = ((c_out * kernel_depth + kd) * kernel_width + kw) * kernel_height + kh
                        w_idx = w_idx * (in_channels // groups) + ci
                        w = tl.load(w_ptr + w_idx)
                        
                        acc += x * w
    
    # Add bias if applicable
    if b_ptr is not None:
        acc += tl.load(b_ptr + c_out)
    
    # Store result
    tl.store(out_ptr + idx, acc)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1, 1),
    padding: tuple = (0, 0, 0),
    output_padding: tuple = (0, 0, 0),
    groups: int = 1
) -> torch.Tensor:
    """
    Custom Triton implementation of ConvTranspose3d.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, depth, width, height = x.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    depth_out = (depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    width_out = (width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1]
    height_out = (height - 1) * stride[2] - 2 * padding[2] + kernel_height + output_padding[2]
    
    # Prepare output tensor
    out = torch.empty((batch_size, out_channels, depth_out, width_out, height_out), dtype=x.dtype, device=x.device)
    
    # Number of elements in the output tensor
    n_elements = out.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        depth, width, height,
        kernel_depth, kernel_width, kernel_height,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        groups,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )