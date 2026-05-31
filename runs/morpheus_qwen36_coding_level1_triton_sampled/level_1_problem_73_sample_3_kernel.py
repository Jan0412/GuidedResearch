import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    in_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    stride,
    padding,
    groups,
    in_channels,
    out_channels,
    kernel_size,
    batch_size,
    in_depth,
    in_height,
    in_width,
    out_depth,
    out_height,
    out_width,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Decode program ID to (n, c_out, d, h)
    num_spatial = out_depth * out_height
    num_c_out = out_channels
    num_n = batch_size
    
    n = pid // (num_c_out * num_spatial)
    rem = pid % (num_c_out * num_spatial)
    c_out = rem // num_spatial
    rem = rem % num_spatial
    d = rem // out_height
    h = rem % out_height
    
    # w offsets within the block
    w_offsets = tl.arange(0, BLOCK_SIZE_W)
    w = w_offsets  # w is relative to block start, which is 0 for this grid mapping
    
    # Mask for w bounds
    w_mask = w < out_width
    
    # Output pointer offset for this (n, c_out, d, h) block
    # out shape: (N, C_out, D_out, H_out, W_out)
    # stride_w_out = 1, stride_h_out = W_out, stride_d_out = H_out * W_out, etc.
    out_base_ptr = (
        n * out_channels * out_depth * out_height * out_width +
        c_out * out_depth * out_height * out_width +
        d * out_height * out_width +
        h * out_width
    )
    out_ptr_block = out_base_ptr + w_offsets
    
    # Input pointer base calculation
    # in shape: (N, C_in, D_in, H_in, W_in)
    # We will compute access for each c_in and kernel element
    
    c_in_per_group = in_channels // groups
    group_idx = c_out // groups
    
    # Load weights for this c_out into registers
    # weight shape: (C_out, C_in/groups, kD, kH, kW)
    # We need weight[c_out, c_in, kd, kh, kw]
    # c_in ranges 0..c_in_per_group-1
    # kd, kh, kw range 0..kernel_size-1
    
    # Precompute weight offsets
    # weight is contiguous in kW, then kH, then kD, then c_in
    # offset = c_in * (kD*kH*kW) + kd * (kH*kW) + kh * kW + kw
    
    # We can load all weights for this c_out into a 1D tensor in registers
    # Total weight elements per c_out: c_in_per_group * kernel_size^3
    num_weight_elems = c_in_per_group * kernel_size * kernel_size * kernel_size
    w_offsets_reg = tl.arange(0, num_weight_elems)
    
    # Weight pointer for this c_out
    weight_base_ptr = weight_ptr + c_out * c_in_per_group * kernel_size * kernel_size * kernel_size
    
    # Load weights
    # We need to map w_offsets_reg to (c_in, kd, kh, kw)
    # This is a bit complex for tl.load directly with broadcasting
    # Instead, we can load in chunks or use tl.reshape if supported
    # Triton supports loading and reshaping
    W = tl.load(weight_base_ptr + w_offsets_reg, mask=w_offsets_reg < num_weight_elems, other=0.0)
    W = tl.reshape(W, (c_in_per_group, kernel_size, kernel_size, kernel_size))
    
    acc = tl.zeros((BLOCK_SIZE_W,), dtype=tl.float32)
    
    # Loop over c_in within group
    for c_in in range(c_in_per_group):
        # Actual input channel index
        c_in_idx = c_in * groups + group_idx
        
        # Input base pointer for this channel
        # in_ptr offset: n * C_in * D_in * H_in * W_in + c_in_idx * D_in * H_in * W_in
        in_base_ptr = (
            n * in_channels * in_depth * in_height * in_width +
            c_in_idx * in_depth * in_height * in_width
        )
        
        # Loop over kernel dimensions
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Compute input coordinates
                    d_in = d - kd * stride + padding
                    h_in = h - kh * stride + padding
                    w_in_base = w - kw * stride + padding
                    
                    # Check spatial bounds for d_in, h_in
                    # w_in is handled by mask and bounds check below
                    
                    d_in_mask = (d_in >= 0) & (d_in < in_depth)
                    h_in_mask = (h_in >= 0) & (h_in < in_height)
                    
                    # If d_in or h_in out of bounds, skip
                    if not d_in_mask or not h_in_mask:
                        continue
                    
                    # w_in varies with w_offsets
                    w_in = w_in_base + w_offsets
                    
                    # w_in bounds
                    w_in_mask = (w_in >= 0) & (w_in < in_width)
                    
                    # Combined mask
                    mask = w_in_mask & w_mask
                    
                    if mask.any():
                        # Load input tile
                        in_ptr_offset = in_base_ptr + d_in * in_height * in_width + h_in * in_width + w_in
                        I = tl.load(in_ptr_offset, mask=mask, other=0.0)
                        
                        # Get weight value
                        w_val = W[c_in, kd, kh, kw]
                        
                        # Accumulate
                        acc += I * w_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c_out)
        acc += bias_val
    
    # Store result
    tl.store(out_ptr_block, acc, mask=w_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, in_depth, in_height, in_width = x.shape
    out_channels, c_in_per_group, kernel_size, _, _ = weight.shape
    
    # Calculate output dimensions
    out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding
    
    out = torch.empty((batch_size, out_channels, out_depth, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Grid: one program per (n, c_out, d, h)
    num_programs = batch_size * out_channels * out_depth * out_height
    BLOCK_SIZE_W = 128  # Tunable block size for width dimension
    
    grid = (num_programs,)
    
    conv_transpose3d_kernel[grid](
        x,
        weight,
        bias,
        out,
        stride,
        padding,
        groups,
        in_channels,
        out_channels,
        kernel_size,
        batch_size,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D transposed convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias,
            stride=self.conv_transpose3d.stride[0],
            padding=self.conv_transpose3d.padding[0],
            output_padding=self.conv_transpose3d.output_padding[0],
            groups=self.conv_transpose3d.groups,
        )