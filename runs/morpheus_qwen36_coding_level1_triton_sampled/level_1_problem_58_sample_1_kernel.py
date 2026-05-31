import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, out_ptr, bias_ptr,
    in_channels, out_channels, kernel_size_d, kernel_size_h, kernel_size_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    output_padding_d, output_padding_h, output_padding_w,
    batch_size, depth_in, height_in, width_in,
    depth_out, height_out, width_out,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid is 1D over output elements
    pid = tl.program_id(0)
    num_elements = batch_size * out_channels * depth_out * height_out * width_out
    
    if pid * BLOCK_SIZE >= num_elements:
        return

    # Offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Compute coordinates for each element in the block
    idx = offsets
    w = idx % width_out
    idx = idx // width_out
    h = idx % height_out
    idx = idx // height_out
    d = idx % depth_out
    idx = idx // depth_out
    c = idx % out_channels
    b = idx // out_channels
    
    # Compute input coordinates base
    # i_d = d * stride_d - padding_d
    # We will add k_d inside the loop
    i_d_base = d * stride_d - padding_d
    i_h_base = h * stride_h - padding_h
    i_w_base = w * stride_h - padding_w  # Note: typo in variable name, should be i_w_base
    
    # Handle output padding
    # Output padding effectively extends the output range
    # The mapping logic remains the same, but we process more elements
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Determine channel groups
    group_size_c = out_channels // groups
    group_idx = c // group_size_c
    group_start_c_in = group_idx * (in_channels // groups)
    group_end_c_in = (group_idx + 1) * (in_channels // groups)
    
    # Loop over input channels and kernel dimensions
    for c_in in range(group_start_c_in, group_end_c_in):
        for k_d in range(kernel_size_d):
            for k_h in range(kernel_size_h):
                for k_w in range(kernel_size_w):
                    # Compute input coordinates
                    i_d = i_d_base + k_d
                    i_h = i_h_base + k_h
                    i_w = i_w_base + k_w
                    
                    # Check bounds
                    i_d_mask = (i_d >= 0) & (i_d < depth_in)
                    i_h_mask = (i_h >= 0) & (i_h < height_in)
                    i_w_mask = (i_w >= 0) & (i_w < width_in)
                    spatial_mask = i_d_mask & i_h_mask & i_w_mask
                    
                    # Compute pointer offsets for x and w
                    # x: (B, C_in, D_in, H_in, W_in)
                    # w: (C_out, C_in, K_d, K_h, K_w)
                    
                    # x offset: b * stride_b + c_in * stride_c + i_d * stride_d + i_h * stride_h + i_w * stride_w
                    # We can compute strides or use tl.linear_to_meshgrid? No, manual is fine.
                    # Strides for x:
                    # stride_b = C_in * D_in * H_in * W_in
                    # stride_c = D_in * H_in * W_in
                    # stride_d = H_in * W_in
                    # stride_h = W_in
                    # stride_w = 1
                    
                    x_offset = (b * (in_channels * depth_in * height_in * width_in) +
                                c_in * (depth_in * height_in * width_in) +
                                i_d * (height_in * width_in) +
                                i_h * width_in +
                                i_w)
                    
                    # w offset: c * stride_c_out + c_in * stride_c_in + k_d * stride_k_d + k_h * stride_k_h + k_w * stride_k_w
                    # w strides:
                    # stride_c_out = C_in * K_d * K_h * K_w
                    # stride_c_in = K_d * K_h * K_w
                    # stride_k_d = K_h * K_w
                    # stride_k_h = K_w
                    # stride_k_w = 1
                    
                    w_offset = (c * (in_channels * kernel_size_d * kernel_size_h * kernel_size_w) +
                                c_in * (kernel_size_d * kernel_size_h * kernel_size_w) +
                                k_d * (kernel_size_h * kernel_size_w) +
                                k_h * kernel_size_w +
                                k_w)
                    
                    # Load values
                    x_val = tl.load(x_ptr + x_offset, mask=spatial_mask & mask, other=0.0)
                    w_val = tl.load(w_ptr + w_offset, mask=mask, other=0.0)
                    
                    acc += x_val * w_val
    
    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c, mask=mask, other=0.0)
        acc += bias_val
    
    # Store result
    tl.store(out_ptr + offsets, acc, mask=mask)


def triton_conv_transpose3d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor,
                            in_channels: int, out_channels: int,
                            kernel_size: tuple, stride: tuple, padding: tuple,
                            output_padding: tuple, groups: int) -> torch.Tensor:
    """
    Wrapper function for the Triton kernel.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, c_in, d_in, h_in, w_in = x.shape
    c_out, c_in_w, k_d, k_h, k_w = w.shape
    
    # Compute output dimensions
    d_out = (d_in - 1) * stride[0] - 2 * padding[0] + k_d + output_padding[0]
    h_out = (h_in - 1) * stride[1] - 2 * padding[1] + k_h + output_padding[1]
    w_out = (w_in - 1) * stride[2] - 2 * padding[2] + k_w + output_padding[2]
    
    out = torch.empty((batch_size, c_out, d_out, h_out, w_out), device=x.device, dtype=x.dtype)
    
    num_elements = batch_size * c_out * d_out * h_out * w_out
    BLOCK_SIZE = 128  # Tunable parameter
    
    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    conv_transpose3d_kernel[grid](
        x, w, out, bias,
        in_channels, out_channels, k_d, k_h, k_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        batch_size, d_in, h_in, w_in,
        d_out, h_out, w_out,
        groups,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            self.in_channels, self.out_channels,
            self.kernel_size, self.stride, self.padding,
            self.output_padding, self.groups
        )


def get_inputs():
    batch_size = 16
    in_channels = 32
    out_channels = 16
    kernel_size = (3, 5, 7)
    depth_in = 16
    height_in = 32
    width_in = 64
    
    x = torch.rand(batch_size, in_channels, depth_in, height_in, width_in).cuda()
    return [x]


def get_init_inputs():
    in_channels = 32
    out_channels = 16
    kernel_size = (3, 5, 7)
    return [in_channels, out_channels, kernel_size]