import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, y_ptr,
    batch_size, in_channels, out_channels,
    height_in, width_in, height_out, width_out,
    kernel_h, kernel_w, stride_h, stride_w,
    padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate global offset for this program
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * out_channels * height_out * width_out

    # Compute output indices
    batch_idx = (offsets // (out_channels * height_out * width_out)) % batch_size
    c_out = (offsets // (height_out * width_out)) % out_channels
    h_out = (offsets // width_out) % height_out
    w_out = offsets % width_out

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over input channels
    for c_in in range(in_channels):
        # Loop over kernel height
        for kh in range(kernel_h):
            # Loop over kernel width
            for kw in range(kernel_w):
                # Calculate corresponding input coordinates
                h_in = h_out * stride_h - padding_h + kh * dilation_h
                w_in = w_out * stride_w - padding_w + kw * dilation_w

                # Check bounds
                h_valid = (h_in >= 0) & (h_in < height_in)
                w_valid = (w_in >= 0) & (w_in < width_in)
                valid = h_valid & w_valid

                # Load weight
                w_offset = c_out * in_channels * kernel_h * kernel_w + c_in * kernel_h * kernel_w + kh * kernel_w + kw
                w = tl.load(w_ptr + w_offset)

                # Load input if valid
                x_offset = ((batch_idx * in_channels + c_in) * height_in + h_in) * width_in + w_in
                x = tl.load(x_ptr + x_offset, mask=valid, other=0.0)

                accumulator += w * x

    # Store result
    tl.store(y_ptr + offsets, accumulator, mask=mask)


def triton_conv_transpose2d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None,
                            stride: tuple = (1, 1), padding: tuple = (0, 0),
                            output_padding: tuple = (0, 0), dilation: tuple = (1, 1)) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_channels, height_in, width_in = x.shape
    out_channels, _, kernel_h, kernel_w = w.shape

    # Calculate output dimensions
    height_out = (height_in - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    width_out = (width_in - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]

    y = torch.empty((batch_size, out_channels, height_out, width_out), dtype=torch.float32, device=x.device)

    num_elements = batch_size * out_channels * height_out * width_out
    BLOCK_SIZE = 128

    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv_transpose2d_kernel[grid](
        x_ptr=x, w_ptr=w, y_ptr=y,
        batch_size=batch_size, in_channels=in_channels, out_channels=out_channels,
        height_in=height_in, width_in=width_in, height_out=height_out, width_out=width_out,
        kernel_h=kernel_h, kernel_w=kernel_w, stride_h=stride[0], stride_w=stride[1],
        padding_h=padding[0], padding_w=padding[1], dilation_h=dilation[0], dilation_w=dilation[1],
        BLOCK_SIZE=BLOCK_SIZE
    )

    if bias is not None:
        y = y + bias.view(1, -1, 1, 1)

    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias
        self.register_parameter('weight', nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1])))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.empty(out_channels)))
        else:
            self.register_parameter('bias', None)

        # Initialize weights using Kaiming uniform for better training dynamics
        nn.init.kaiming_uniform_(self.weight, a=0.0)
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, dilation=self.dilation
        )