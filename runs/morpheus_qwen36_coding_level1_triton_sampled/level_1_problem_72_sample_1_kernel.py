import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    in_ptr, weight_ptr, bias_ptr, out_ptr,
    in_strides, weight_strides, out_strides,
    N, C_in, C_out, D_in, H_in, W_in,
    D_out, H_out, W_out,
    K_d, K_h, K_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    groups, has_bias,
    BLOCK_SIZE: tl.constexpr
):
    # Decode program IDs for each output element
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    d = tl.program_id(2)
    h = tl.program_id(3)
    w = tl.program_id(4)

    # Determine group for current output channel
    c_in_per_group = C_in // groups
    group = c_out // c_in_per_group
    c_in_start = group * c_in_per_group

    # Reduction dimensions
    num_k = K_d * K_h * K_w
    total_red = c_in_per_group * num_k

    acc = 0.0

    # Loop over reduction dimensions with blocking
    for r in range(0, total_red, BLOCK_SIZE):
        offsets = r + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_red

        # Decode reduction index into c_in and kernel index
        c_in_idx = offsets // num_k
        k_idx = offsets % num_k

        k_d = k_idx // (K_h * K_w)
        k_h = (k_idx // K_w) % K_h
        k_w = k_idx % K_w

        # Compute input coordinates
        in_d = d * stride_d - pad_d + k_d
        in_h = h * stride_h - pad_h + k_h
        in_w = w * stride_w - pad_w + k_w

        # Actual input channel index
        c_in_actual = c_in_start + c_in_idx

        # Compute input pointer offset
        in_offset = (n * in_strides[0] +
                     c_in_actual * in_strides[1] +
                     in_d * in_strides[2] +
                     in_h * in_strides[3] +
                     in_w * in_strides[4])

        # Mask for valid input coordinates
        in_mask = (in_d >= 0) & (in_d < D_in) & \
                  (in_h >= 0) & (in_h < H_in) & \
                  (in_w >= 0) & (in_w < W_in)

        # Load input values (0.0 for out-of-bounds)
        in_vals = tl.load(in_ptr + in_offset, mask=in_mask, other=0.0)

        # Compute weight pointer offset
        weight_offset = (c_out * weight_strides[0] +
                         c_in_actual * weight_strides[1] +
                         k_d * weight_strides[2] +
                         k_h * weight_strides[3] +
                         k_w * weight_strides[4])

        # Load weight values (always valid due to mask on offsets)
        w_vals = tl.load(weight_ptr + weight_offset)

        # Accumulate product
        acc += tl.sum(in_vals * w_vals)

    # Store result
    out_offset = (n * out_strides[0] +
                  c_out * out_strides[1] +
                  d * out_strides[2] +
                  h * out_strides[3] +
                  w * out_strides[4])

    out_val = acc

    if has_bias:
        bias_val = tl.load(bias_ptr + c_out)
        out_val += bias_val

    tl.store(out_ptr + out_offset, out_val)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups, has_bias):
    """
    Wrapper function to launch the custom Triton kernel for 3D transposed convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, K_d, K_h, K_w = weight.shape

    # Compute output dimensions matching PyTorch's ConvTranspose3d
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + K_d + output_padding[0]
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + K_h + output_padding[1]
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + K_w + output_padding[2]

    out = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides
    in_strides = x.stride()
    weight_strides = weight.stride()
    out_strides = out.stride()

    # Grid configuration: one program per output element
    grid = (N, C_out, D_out, H_out, W_out)

    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        in_strides, weight_strides, out_strides,
        N, C_in, C_out, D_in, H_in, W_in,
        D_out, H_out, W_out,
        K_d, K_h, K_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        groups, has_bias,
        BLOCK_SIZE=128
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized 3D transposed convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1),
                 padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters (Xavier uniform similar to PyTorch default)
        nn.init.xavier_uniform_(self.weight)
        if bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using the custom Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding,
            self.groups, self.has_bias
        )