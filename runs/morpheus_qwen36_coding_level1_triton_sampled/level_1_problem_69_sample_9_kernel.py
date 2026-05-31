import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_height,
    in_width,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    dilation_height,
    dilation_width,
    output_height,
    output_width,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Grid dimensions
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)

    # Block offsets
    block_h_start = out_h_idx * BLOCK_SIZE_Y
    block_w_start = out_w_idx * BLOCK_SIZE_X

    # Create offsets for output tile
    h_offsets = block_h_start + tl.arange(0, BLOCK_SIZE_Y)
    w_offsets = block_w_start + tl.arange(0, BLOCK_SIZE_X)

    # Mask for output tile bounds
    h_mask = h_offsets < output_height
    w_mask = w_offsets < output_width
    mask_2d = h_mask[:, None] & w_mask[None, :]

    # Accumulator for the output tile
    acc = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)

    # Precompute base input coordinates for this output tile
    # Input row index for output row h: h * stride_h - padding_h
    base_in_row = h_offsets * stride_height - padding_height
    base_in_col = w_offsets * stride_width - padding_width

    # Load weights for the current output channel
    # Weights shape: (out_channels, in_channels, kernel_height, kernel_width)
    # We need weights for out_ch_idx, all in_channels, kernel dimensions
    w_ptr_ch = w_ptr + out_ch_idx * in_channels * kernel_height * kernel_width
    w = tl.load(w_ptr_ch + tl.arange(0, in_channels * BLOCK_SIZE_KH * BLOCK_SIZE_KW).reshape(in_channels, BLOCK_SIZE_KH, BLOCK_SIZE_KW),
                mask=None, other=0.0)

    # Loop over input channels
    for in_ch in range(in_channels):
        # Load input tile into shared memory or compute directly
        # Since input access is strided, we compute indices on the fly
        # Input tile coordinates depend on kernel dilation and size
        # For each kernel element (kh, kw), input index is base_in_row + kh * dilation_h, base_in_col + kw * dilation_w
        
        # We can vectorize over kernel dimensions
        kh_offsets = tl.arange(0, BLOCK_SIZE_KH)
        kw_offsets = tl.arange(0, BLOCK_SIZE_KW)
        
        # Compute input row and column offsets for all kernel elements
        # These are broadcasted with h_offsets and w_offsets
        in_row_offsets = base_in_row[:, None] + kh_offsets[None, :] * dilation_height
        in_col_offsets = base_in_col[:, None] + kw_offsets[None, :] * dilation_width
        
        # Create masks for input bounds
        in_row_mask = (in_row_offsets >= 0) & (in_row_offsets < in_height)
        in_col_mask = (in_col_offsets >= 0) & (in_col_offsets < in_width)
        in_mask = in_row_mask & in_col_mask
        
        # Load input values
        # Input shape: (batch_size, in_channels, in_height, in_width)
        # Stride: in_channels * in_height * in_width, in_height * in_width, in_width, 1
        x_ptr_ch = x_ptr + batch_idx * in_channels * in_height * in_width + in_ch * in_height * in_width
        x_tile = tl.load(x_ptr_ch + in_row_offsets * in_width + in_col_offsets,
                         mask=in_mask, other=0.0)
        
        # Multiply with weights and accumulate
        # w shape: (in_channels, BLOCK_SIZE_KH, BLOCK_SIZE_KW)
        # x_tile shape: (BLOCK_SIZE_Y, BLOCK_SIZE_X)
        # We need to multiply x_tile with w[in_ch, :, :]
        # w_tile shape: (BLOCK_SIZE_KH, BLOCK_SIZE_KW)
        w_tile = w[in_ch, :, :]
        acc += x_tile * w_tile

    # Store result
    out_ptr_offset = out_ptr + batch_idx * out_channels * output_height * output_width + out_ch_idx * output_height * output_width
    tl.store(out_ptr_offset + h_offsets[:, None] * output_width + w_offsets[None, :], acc, mask=mask_2d)


def triton_conv_transpose2d(x: torch.Tensor, w: torch.Tensor, stride: tuple, padding: tuple, output_padding: tuple, dilation: tuple) -> torch.Tensor:
    """
    Custom Triton kernel for ConvTranspose2d.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_height, kernel_width = w.shape
    
    # Calculate output dimensions
    # Formula: (in_size - 1) * stride - 2 * padding + kernel_size + output_padding
    out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0]
    out_width = (in_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1]
    
    out = torch.empty((batch_size, out_channels, out_height, out_width), dtype=x.dtype, device=x.device)
    
    # Tunable block sizes
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    BLOCK_SIZE_KH = 3
    BLOCK_SIZE_KW = 3
    
    # Grid configuration
    grid = (batch_size, out_channels, 
            (out_height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y, 
            (out_width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X)
    
    conv_transpose2d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_height, kernel_width,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        out_height, out_width,
        BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_KH, BLOCK_SIZE_KW
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for ConvTranspose2d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Note: groups > 1 is not fully optimized in this kernel for simplicity,
        # assuming groups=1 for the custom implementation to focus on the core transpose logic.
        # For groups > 1, a more complex kernel or fallback would be needed.
        # Here we assume groups=1 as per the example constraints and common optimization targets.
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose2d.weight
        # If bias exists, add it after convolution
        if self.conv_transpose2d.bias is not None:
            bias = self.conv_transpose2d.bias
        else:
            bias = None
            
        out = triton_conv_transpose2d(x, w, self.stride, self.padding, self.output_padding, self.dilation)
        
        if bias is not None:
            # Add bias: shape (out_channels, 1, 1) broadcastable
            out = out + bias.view(1, -1, 1, 1)
            
        return out