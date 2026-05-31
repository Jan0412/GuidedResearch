import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    batch_size,
    in_channels,
    out_channels,
    in_depth, in_height, in_width,
    out_depth, out_height, out_width,
    kernel_depth, kernel_height, kernel_width,
    stride_depth, stride_height, stride_width,
    padding_depth, padding_height, padding_width,
    output_padding_depth, output_padding_height, output_padding_width,
    groups,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Calculate output coordinates
    pid = tl.program_id(0)
    n = pid // (out_channels * out_depth * out_height * out_width)
    rem = pid % (out_channels * out_depth * out_height * out_width)
    c_out = rem // (out_depth * out_height * out_width)
    rem = rem % (out_depth * out_height * out_width)
    d = rem // (out_height * out_width)
    h = rem // out_width
    w = rem % out_width

    # Calculate input coordinates base
    # Formula: out = in * stride - padding + kernel - 1 + output_padding
    # So in = (out + padding - kernel + 1 - output_padding) / stride
    # But we iterate kernel indices and compute input indices
    # out_idx = in_idx * stride + (kernel - 1 - padding - output_padding)
    # in_idx = (out_idx - (kernel - 1 - padding - output_padding)) / stride
    
    # We can compute the offset for input index
    offset_d = kernel_depth - 1 - padding_depth - output_padding_depth
    offset_h = kernel_height - 1 - padding_height - output_padding_height
    offset_w = kernel_width - 1 - padding_width - output_padding_width

    # Accumulator
    acc = 0.0

    # Determine channel group
    channels_per_group = in_channels // groups
    group_id = c_out // (out_channels // groups)
    group_start = group_id * channels_per_group
    group_end = group_start + channels_per_group

    # Loop over input channels in the group
    for c_in_off in tl.arange(0, BLOCK_SIZE_C):
        c_in = group_start + c_in_off
        if c_in < in_channels:
            # Loop over kernel dimensions
            for kd_off in tl.arange(0, BLOCK_SIZE_KD):
                kd = kd_off
                if kd < kernel_depth:
                    for kh_off in tl.arange(0, BLOCK_SIZE_KH):
                        kh = kh_off
                        if kh < kernel_height:
                            for kw_off in tl.arange(0, BLOCK_SIZE_KW):
                                kw = kw_off
                                if kw < kernel_width:
                                    # Calculate input coordinates
                                    di = d * stride_depth - offset_d + kd
                                    dh = h * stride_height - offset_h + kh
                                    dw = w * stride_width - offset_w + kw

                                    # Check bounds
                                    if di >= 0 and di < in_depth and dh >= 0 and dh < in_height and dw >= 0 and dw < in_width:
                                        # Load weights
                                        w_idx = c_out * (in_channels * kernel_depth * kernel_height * kernel_width) + \
                                                c_in * (kernel_depth * kernel_height * kernel_width) + \
                                                kd * (kernel_height * kernel_width) + \
                                                kh * kernel_width + kw
                                        w_val = tl.load(w_ptr + w_idx)
                                        
                                        # Load input
                                        x_idx = n * (in_channels * in_depth * in_height * in_width) + \
                                                c_in * (in_depth * in_height * in_width) + \
                                                di * (in_height * in_width) + \
                                                dh * in_width + dw
                                        x_val = tl.load(x_ptr + x_idx)
                                        
                                        acc += x_val * w_val

    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + c_out)
        acc += b_val

    # Store output
    out_idx = n * (out_channels * out_depth * out_height * out_width) + \
              c_out * (out_depth * out_height * out_width) + \
              d * (out_height * out_width) + \
              h * out_width + w
    tl.store(out_ptr + out_idx, acc)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Custom Triton implementation of ConvTranspose3d.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, in_depth, in_height, in_width = x.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kernel_height + output_padding[1]
    out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kernel_width + output_padding[2]
    
    out = torch.empty((batch_size, out_channels, out_depth, out_height, out_width), device=x.device, dtype=x.dtype)
    
    n_elements = batch_size * out_channels * out_depth * out_height * out_width
    
    # Tunable block sizes
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_KD = 3
    BLOCK_SIZE_KH = 5
    BLOCK_SIZE_KW = 7
    
    # Grid configuration
    grid = (n_elements,)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out, n_elements,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_depth, kernel_height, kernel_width,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        groups,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_KD=BLOCK_SIZE_KD,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D transposed convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using custom Triton kernel.
        """
        return triton_conv_transpose3d(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else None,
            self.conv_transpose3d.stride,
            self.conv_transpose3d.padding,
            self.conv_transpose3d.output_padding,
            self.conv_transpose3d.groups
        )


def get_inputs():
    batch_size = 8
    in_channels = 32
    depth = 12
    height = 24
    width = 48
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    in_channels = 32
    out_channels = 32
    kernel_size = (3, 5, 7)
    stride = (2, 2, 2)
    padding = (1, 2, 3)
    output_padding = (1, 1, 1)
    groups = 4
    bias = False
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias]