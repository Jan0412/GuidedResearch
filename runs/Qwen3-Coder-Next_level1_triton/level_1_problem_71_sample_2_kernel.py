import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,                # Input tensor pointer (B, C_in, H_in, W_in)
    w_ptr,                # Weight tensor pointer (C_in, C_out, K_h, K_w)
    b_ptr,                # Bias tensor pointer (C_out,) or None
    out_ptr,              # Output tensor pointer (B, C_out, H_out, W_out)
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    K_h: tl.constexpr,
    K_w: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    groups: tl.constexpr,
    # Block sizes for tiling
    BLOCK_SIZE_B: tl.constexpr = 1,
    BLOCK_SIZE_C_OUT: tl.constexpr = 16,
    BLOCK_SIZE_C_IN: tl.constexpr = 16,
    BLOCK_SIZE_H: tl.constexpr = 8,
    BLOCK_SIZE_W: tl.constexpr = 8,
):
    # Batch and output channel indices
    batch_idx = tl.program_id(0)
    c_out_block = tl.program_id(1) * BLOCK_SIZE_C_OUT
    h_block = tl.program_id(2) * BLOCK_SIZE_H
    w_block = tl.program_id(3) * BLOCK_SIZE_W
    
    # Create ranges for output channel, height, and width
    c_out_offsets = c_out_block + tl.arange(0, BLOCK_SIZE_C_OUT)
    h_offsets = h_block + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_block + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid indices
    c_out_mask = c_out_offsets < C_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    
    # Compute the actual output positions
    h_out = h_offsets[None, :, None]
    w_out = w_offsets[None, None, :]
    c_out = c_out_offsets[:, None, None]
    
    # Compute corresponding input positions for transposed convolution
    # For transposed conv: input_h = (output_h - output_padding - (kernel_h - 1) + padding) // stride + 1
    # But we iterate over input positions instead
    # Actually, for transposed conv: output_h = (input_h - 1) * stride - 2*padding + output_padding + kernel_h
    # So input_h = (output_h - output_padding - kernel_h + 1 + 2*padding) // stride + 1
    
    # Instead, we'll iterate over input positions and accumulate to output
    # Let's restructure: for each output position, sum over input positions and kernel
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C_OUT, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_block in range(0, C_in, BLOCK_SIZE_C_IN):
        c_in_offsets = c_in_block + tl.arange(0, BLOCK_SIZE_C_IN)
        c_in_mask = c_in_offsets < C_in
        
        # Load input block: shape (B, C_in, H_in, W_in)
        # For this implementation, we'll iterate over input spatial positions and accumulate to output
        # But this is inefficient - better to iterate output positions and compute contributions
        
    # Alternative approach: for each output position, compute contribution from all input positions and kernel
    # We'll restructure the kernel to be more efficient
    
    # Let's use a different approach: iterate over output positions
    # For each output position (b, c_out, h_out, w_out), compute:
    # out[b, c_out, h_out, w_out] = sum_{c_in, k_h, k_w} x[b, c_in, h_in, w_in] * w[c_in, c_out, k_h, k_w]
    # where h_in = (h_out - output_padding - k_h + stride) // stride, w_in = (w_out - output_padding - k_w + stride) // stride
    
    # We'll do this with nested loops for better memory access patterns
    
    # Get the output position offsets
    c_out_offsets = c_out_block + tl.arange(0, BLOCK_SIZE_C_OUT)
    h_offsets = h_block + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_block + tl.arange(0, BLOCK_SIZE_W)
    
    # Create meshgrid for output positions
    c_out_i, h_i, w_i = tl.meshgrid(c_out_offsets, h_offsets, w_offsets)
    c_out_i = c_out_i.T
    h_i = h_i.T
    w_i = w_i.T
    
    # Masks
    c_out_mask_i = c_out_i < C_out
    h_mask_i = h_i < H_out
    w_mask_i = w_i < W_out
    valid_mask = c_out_mask_i & h_mask_i & w_mask_i
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C_OUT), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_block in range(0, C_in, BLOCK_SIZE_C_IN):
        c_in_offsets = c_in_block + tl.arange(0, BLOCK_SIZE_C_IN)
        c_in_mask = c_in_offsets < C_in
        
        # Loop over kernel height
        for k_h in range(K_h):
            # Compute input height for this kernel position
            h_in = (h_i - k_h + padding) // stride
            h_in_valid = (h_in >= 0) & (h_in < H_in)
            
            # Loop over kernel width
            for k_w in range(K_w):
                # Compute input width for this kernel position
                w_in = (w_i - k_w + padding) // stride
                w_in_valid = (w_in >= 0) & (w_in < W_in)
                
                # Combined valid mask
                in_valid = h_in_valid & w_in_valid & valid_mask
                
                # Load input values
                # x_ptr offset: b * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in
                x_offsets = batch_idx * (C_in * H_in * W_in) + c_in_offsets[:, None, None, None] * (H_in * W_in) + \
                           h_in[None, :, :, :] * W_in + w_in[None, :, :, :]
                x_offsets = x_offsets.transpose(0, 3, 1, 2)  # (C_in_block, B, H, W) -> we need to handle this carefully
                
                # Simplified approach: iterate over C_in_block
                for c_in_idx in range(BLOCK_SIZE_C_IN):
                    if c_in_block + c_in_idx < C_in:
                        x_offset = batch_idx * (C_in * H_in * W_in) + (c_in_block + c_in_idx) * (H_in * W_in) + \
                                  h_in * W_in + w_in
                        x_val = tl.load(x_ptr + x_offset, mask=in_valid, other=0.0)
                        
                        # Load kernel values: w[c_in, c_out, k_h, k_w]
                        # w_ptr offset: c_in * (C_out * K_h * K_w) + c_out * (K_h * K_w) + k_h * K_w + k_w
                        w_offset = (c_in_block + c_in_idx) * (C_out * K_h * K_w) + c_out_i * (K_h * K_w) + \
                                  k_h * K_w + k_w
                        w_val = tl.load(w_ptr + w_offset, mask=in_valid, other=0.0)
                        
                        # Accumulate
                        acc += x_val[:, :, :] * w_val[:, :, :]
    
    # Add bias if present
    if b_ptr is not None:
        b = tl.load(b_ptr + c_out_i, mask=c_out_mask_i, other=0.0)
        acc += b[None, None, :]
    
    # Store result
    out_offset = batch_idx * (C_out * H_out * W_out) + c_out_i * (H_out * W_out) + h_i * W_out + w_i
    tl.store(out_ptr + out_offset, acc.T, mask=valid_mask)


def triton_conv_transpose2d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose2d
    
    Args:
        x: Input tensor of shape (B, C_in, H_in, W_in)
        weight: Weight tensor of shape (C_in, C_out, K_h, K_w)
        bias: Bias tensor of shape (C_out,) or None
        stride, padding, output_padding, groups: Convolution parameters
    """
    B, C_in, H_in, W_in = x.shape
    _, C_out, K_h, K_w = weight.shape
    
    # Compute output shape
    H_out = (H_in - 1) * stride - 2 * padding + output_padding + K_h
    W_out = (W_in - 1) * stride - 2 * padding + output_padding + K_w
    
    # Create output tensor
    out = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Grid configuration
    # We'll use a 4D grid: [batch, c_out_blocks, h_blocks, w_blocks]
    BLOCK_SIZE_C_OUT = 8
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    grid = lambda meta: (
        B,
        (C_out + meta['BLOCK_SIZE_C_OUT'] - 1) // meta['BLOCK_SIZE_C_OUT'],
        (H_out + meta['BLOCK_SIZE_H'] - 1) // meta['BLOCK_SIZE_H'],
        (W_out + meta['BLOCK_SIZE_W'] - 1) // meta['BLOCK_SIZE_W'],
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        B=B, C_in=C_in, C_out=C_out,
        H_in=H_in, W_in=W_in,
        K_h=K_h, K_w=K_w,
        H_out=H_out, W_out=W_out,
        stride=stride, padding=padding, output_padding=output_padding, groups=groups,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as original
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create the weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) / (in_channels * kernel_size * kernel_size) ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )