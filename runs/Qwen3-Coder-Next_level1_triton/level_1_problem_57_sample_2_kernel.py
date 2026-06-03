import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv2d_kernel(
    # Pointers to tensors
    x_ptr,  # Input: (B, C_in, H_in, W_in)
    w_ptr,  # Weight: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias: (C_out,) or None
    y_ptr,  # Output: (B, C_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, H_in, W_in,
    C_out, K_h, K_w,
    H_out, W_out,
    # Strides
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get batch, output channel, output height, output width indices
    batch_idx = tl.program_id(0)
    c_out_block = tl.program_id(1)
    h_out_block = tl.program_id(2)
    w_out_block = tl.program_id(3)
    
    # Calculate actual output channel and spatial indices
    c_out = c_out_block * BLOCK_SIZE_COUT
    h_out = h_out_block * BLOCK_SIZE_COUT  # Note: using same block size for simplicity
    w_out = w_out_block * BLOCK_SIZE_COUT
    
    # Create ranges for output channels, input channels, and kernel dimensions
    c_out_offsets = c_out + tl.arange(0, BLOCK_SIZE_COUT)
    c_in_offsets = tl.arange(0, BLOCK_SIZE_CIN)
    k_h_offsets = tl.arange(0, BLOCK_SIZE_KH)
    k_w_offsets = tl.arange(0, BLOCK_SIZE_KW)
    
    # Create masks for valid indices
    c_out_mask = c_out_offsets < C_out
    c_in_mask = c_in_offsets < C_in
    k_h_mask = k_h_offsets < K_h
    k_w_mask = k_w_offsets < K_w
    
    # Initialize accumulator for output
    output = tl.zeros((BLOCK_SIZE_COUT,), tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in_idx in range(0, C_in, BLOCK_SIZE_CIN):
        for k_h_idx in range(0, K_h, BLOCK_SIZE_KH):
            for k_w_idx in range(0, K_w, BLOCK_SIZE_KW):
                # Calculate corresponding input position
                # For transposed convolution: h_in = h_out - (k_h - 1) + padding - stride * (h_out - 1)
                # Actually, the relationship is: h_out = (h_in - 1) * stride - 2 * padding + (k_h - 1) + output_padding + 1
                # So h_in = (h_out - 1 - output_padding) // stride + 1 + padding - (k_h - 1) // stride
                
                # More precise calculation:
                # h_in * stride - padding + (k_h - 1) >= h_out >= h_in * stride - padding
                # => h_in = (h_out + padding - (k_h - 1) + stride - 1) // stride
                # But we need to iterate over k_h and k_w to find valid h_in, w_in
                
                # For each output position (h_out, w_out), we sum over:
                # input positions: h_in = h_out - (k_h - 1) + padding - stride * (h_out - 1) ... wait
                # Actually: h_out = (h_in - 1) * stride - 2 * padding + (k_h - 1) + output_padding + 1
                # So for fixed h_out and k_h, h_in = (h_out + 2*padding - (k_h - 1) - output_padding) // stride + 1
                
                # Let's compute h_in and w_in for each (k_h, k_w) that contributes to (h_out, w_out)
                h_in = h_out + padding - k_h_idx
                w_in = w_out + padding - k_w_idx
                
                # Check if this input position is valid
                valid_input = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
                
                if valid_input:
                    # Load input value
                    x_offset = batch_idx * (C_in * H_in * W_in) + c_in_idx * (H_in * W_in) + h_in * W_in + w_in
                    x_val = tl.load(x_ptr + x_offset, mask=c_in_mask, other=0.0)
                    
                    # Load weight value
                    w_offset = c_in_idx * (C_out * K_h * K_w) + c_out_offsets[:, None, None] * (K_h * K_w) + k_h_idx * K_w + k_w_idx
                    w_val = tl.load(w_ptr + w_offset, mask=c_in_mask[:, None, None] & c_out_mask[None, :, None] & k_h_mask[None, None, :] & k_w_mask[None, None, :], other=0.0)
                    
                    # Accumulate: output[c_out] += x[h_in, w_in] * w[c_in, c_out, k_h, k_w]
                    output += x_val[:, None, None] * w_val
    
    # Handle bias if present
    if b_ptr is not None:
        b = tl.load(b_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
        output += b
    
    # Store result
    y_offset = batch_idx * (C_out * H_out * W_out) + c_out_offsets * (H_out * W_out) + h_out * W_out + w_out
    tl.store(y_ptr + y_offset, output, mask=c_out_mask & (h_out < H_out) & (w_out < W_out))


# Better approach: implement transposed conv as regular conv with padded input
# Actually, let's use a more efficient approach with tiling and better memory access

@triton.jit
def transposed_conv2d_kernel_optimized(
    x_ptr,  # Input: (B, C_in, H_in, W_in)
    w_ptr,  # Weight: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias: (C_out,)
    y_ptr,  # Output: (B, C_out, H_out, W_out)
    # Dimensions
    B, C_in, H_in, W_in,
    C_out, K_h, K_w,
    H_out, W_out,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    c_out_block = tl.program_id(1)
    h_block = tl.program_id(2)
    w_block = tl.program_id(3)
    
    # Calculate output positions
    c_out = c_out_block * BLOCK_SIZE_COUT
    h_out_start = h_block * BLOCK_SIZE_H
    w_out_start = w_block * BLOCK_SIZE_W
    
    # Create offsets and masks
    c_out_offsets = c_out + tl.arange(0, BLOCK_SIZE_COUT)
    c_out_mask = c_out_offsets < C_out
    
    h_out_offsets = h_out_start + tl.arange(0, BLOCK_SIZE_H)
    w_out_offsets = w_out_start + tl.arange(0, BLOCK_SIZE_W)
    
    h_out_mask = h_out_offsets < H_out
    w_out_mask = w_out_offsets < W_out
    
    # Initialize output accumulator
    output_acc = tl.zeros((BLOCK_SIZE_COUT, BLOCK_SIZE_H, BLOCK_SIZE_W), tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in_idx in range(0, C_in, BLOCK_SIZE_CIN):
        c_in_offsets = c_in_idx + tl.arange(0, BLOCK_SIZE_CIN)
        c_in_mask = c_in_offsets < C_in
        
        for k_h in range(0, K_h, BLOCK_SIZE_KH):
            k_h_offsets = k_h + tl.arange(0, BLOCK_SIZE_KH)
            k_h_mask = k_h_offsets < K_h
            
            for k_w in range(0, K_w, BLOCK_SIZE_KW):
                k_w_offsets = k_w + tl.arange(0, BLOCK_SIZE_KW)
                k_w_mask = k_w_offsets < K_w
                
                # For each output position, calculate corresponding input position
                # In transposed conv: h_out = (h_in - 1) * stride - 2*padding + (k_h - 1) + output_padding + 1
                # Rearranged: h_in = (h_out + 2*padding - (k_h - 1) - output_padding) / stride + 1
                # But we need to compute for each (h_out, k_h) what h_in would be
                
                # Actually, the mathematical definition is:
                # y[b, c_out, h_out, w_out] = sum_{c_in, k_h, k_w} x[b, c_in, h_in, w_in] * w[c_in, c_out, k_h, k_w]
                # where h_in = h_out - (K_h - 1 - k_h) * stride + padding - output_padding // 2 (simplified)
                
                # More accurately, for transposed conv:
                # h_in = h_out - (k_h - (K_h - 1)) * stride + padding
                # But standard implementation uses: h_in = h_out - k_h * stride + (K_h - 1) * stride + padding
                
                # Let's use the correct formula: 
                # For a transposed convolution, the output at (h_out, w_out) receives contributions from input positions
                # where h_in = (h_out - (K_h - 1 - k_h) * stride + padding - output_padding) / stride
                # This is getting complex, let's use a simpler approach
                
                # For each output position, find which input positions contribute
                for h_out_idx in range(BLOCK_SIZE_H):
                    h_out_val = h_out_offsets[h_out_idx]
                    h_in = h_out_val + k_h - (K_h - 1) * stride - padding
                    
                    if h_in >= 0 and h_in < H_in:
                        for w_out_idx in range(BLOCK_SIZE_W):
                            w_out_val = w_out_offsets[w_out_idx]
                            w_in = w_out_val + k_w - (K_w - 1) * stride - padding
                            
                            if w_in >= 0 and w_in < W_in:
                                # Load input
                                x_offset = (batch_id * C_in * H_in * W_in + 
                                           c_in_offsets[:, None, None] * H_in * W_in + 
                                           h_in * W_in + w_in)
                                x_val = tl.load(x_ptr + x_offset, 
                                               mask=c_in_mask[:, None, None], 
                                               other=0.0)
                                
                                # Load weights
                                w_offset = (c_in_offsets[:, None, None] * C_out * K_h * K_w +
                                           c_out_offsets[None, :, None] * K_h * K_w +
                                           k_h * K_w + k_w)
                                w_val = tl.load(w_ptr + w_offset,
                                               mask=c_in_mask[:, None, None] & 
                                                    c_out_mask[None, :, None] &
                                                    k_h_mask[None, None, :] &
                                                    k_w_mask[None, None, :],
                                               other=0.0)
                                
                                # Accumulate
                                output_acc += x_val[:, :, None, None] * w_val[None, :, :, :]
    
    # Add bias
    if b_ptr is not None:
        b = tl.load(b_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
        output_acc += b[:, None, None]
    
    # Store result
    for h_out_idx in range(BLOCK_SIZE_H):
        h_out_val = h_out_offsets[h_out_idx]
        for w_out_idx in range(BLOCK_SIZE_W):
            w_out_val = w_out_offsets[w_out_idx]
            if h_out_val < H_out and w_out_val < W_out:
                y_offset = (batch_id * C_out * H_out * W_out +
                           c_out_offsets * H_out * W_out +
                           h_out_val * W_out + w_out_val)
                tl.store(y_ptr + y_offset, output_acc[:, h_out_idx, w_out_idx], 
                        mask=c_out_mask)


# Even better: implement using the standard transposed convolution formula with proper indexing
@triton.jit
def transposed_conv2d_fused_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C_in, H_in, W_in,
    C_out, K_h, K_w,
    H_out, W_out,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Program indices
    b_idx = tl.program_id(0)
    c_out_block = tl.program_id(1)
    h_block = tl.program_id(2)
    w_block = tl.program_id(3)
    
    # Output positions
    c_out = c_out_block * BLOCK_SIZE_COUT
    h_start = h_block * BLOCK_SIZE_H
    w_start = w_block * BLOCK_SIZE_W
    
    # Output offsets
    c_out_offsets = c_out + tl.arange(0, BLOCK_SIZE_COUT)
    c_out_mask = c_out_offsets < C_out
    
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    
    # Initialize accumulator
    output = tl.zeros((BLOCK_SIZE_COUT, BLOCK_SIZE_H, BLOCK_SIZE_W), tl.float32)
    
    # Iterate over input channels
    for c_in in range(0, C_in, BLOCK_SIZE_CIN):
        c_in_offsets = c_in + tl.arange(0, BLOCK_SIZE_CIN)
        c_in_mask = c_in_offsets < C_in
        
        # Iterate over kernel height
        for kh in range(0, K_h, BLOCK_SIZE_KH):
            kh_offsets = kh + tl.arange(0, BLOCK_SIZE_KH)
            kh_mask = kh_offsets < K_h
            
            # Iterate over kernel width
            for kw in range(0, K_w, BLOCK_SIZE_KW):
                kw_offsets = kw + tl.arange(0, BLOCK_SIZE_KW)
                kw_mask = kw_offsets < K_w
                
                # For each output position, compute contribution
                for h_idx in range(BLOCK_SIZE_H):
                    h_out = h_offsets[h_idx]
                    # h_in = (h_out - (K_h - 1 - kh) * stride + padding - output_padding) / stride
                    # Simplified: h_in = h_out - kh * stride + (K_h - 1) * stride - padding
                    # Actually, the correct relation is:
                    # h_out = (h_in - 1) * stride - 2*padding + (kh - 1) + output_padding + 1
                    # => h_in = (h_out + 2*padding - (kh - 1) - output_padding) / stride + 1
                    
                    # Let's use the standard transposed conv formula:
                    # For a transposed convolution with kernel position kh, the contribution to h_out comes from:
                    # h_in = h_out - (K_h - 1 - kh) * stride + padding
                    # But this varies by implementation. Let's use the most common definition:
                    # h_in = (h_out - (K_h - 1 - kh) * stride + padding - output_padding) / stride
                    
                    # Actually, PyTorch's implementation computes:
                    # h_in = h_out - kh * stride + (K_h - 1) * stride - padding
                    # But we need to be more precise
                    
                    # Let's compute the valid input positions for this (h_out, kh)
                    # h_in = (h_out + padding - kh + (K_h - 1) * stride) / stride
                    # No, let's use the direct formula from PyTorch docs:
                    # h_out = (h_in - 1) * stride - 2 * padding + (kh - 1) + output_padding + 1
                    # => h_in = (h_out + 2*padding - (kh - 1) - output_padding) / stride + 1
                    
                    # For simplicity, let's check each possible h_in and see if it maps to h_out
                    # This is inefficient but correct. Let's optimize:
                    
                    # The input position h_in contributes to output position:
                    # h_out = h_in * stride - padding + kh
                    # => h_in = (h_out + padding - kh) / stride
                    
                    h_in = (h_out + padding - kh) // stride
                    w_in = (w_offsets + padding - kw) // stride
                    
                    # Check validity
                    h_in_valid = (h_in >= 0) & (h_in < H_in) & ((h_out + padding - kh) % stride == 0)
                    w_in_valid = (w_in >= 0) & (w_in < W_in) & ((w_out_offsets + padding - kw) % stride == 0)
                    valid = h_in_valid & w_in_valid
                    
                    if tl.sum(valid) > 0:
                        # Load input
                        x_offset = (b_idx * C_in * H_in * W_in +
                                   c_in_offsets[:, None, None] * H_in * W_in +
                                   h_in * W_in + w_in)
                        x_val = tl.load(x_ptr + x_offset, 
                                       mask=c_in_mask[:, None, None] & valid[None, :, :], 
                                       other=0.0)
                        
                        # Load weights
                        w_offset = (c_in_offsets[:, None, None] * C_out * K_h * K_w +
                                   c_out_offsets[None, :, None] * K_h * K_w +
                                   kh * K_w + kw)
                        w_val = tl.load(w_ptr + w_offset,
                                       mask=c_in_mask[:, None, None] & 
                                            c_out_mask[None, :, None] &
                                            kh_mask[None, None, :] &
                                            kw_mask[None, None, :],
                                       other=0.0)
                        
                        # Accumulate
                        output += x_val[:, :, None, None] * w_val[None, :, :, :]
    
    # Add bias
    if b_ptr is not None:
        b = tl.load(b_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
        output += b[:, None, None]
    
    # Store result
    for h_idx in range(BLOCK_SIZE_H):
        h_out_val = h_offsets[h_idx]
        for w_idx in range(BLOCK_SIZE_W):
            w_out_val = w_offsets[w_idx]
            if h_out_val < H_out and w_out_val < W_out:
                y_offset = (b_idx * C_out * H_out * W_out +
                           c_out_offsets * H_out * W_out +
                           h_out_val * W_out + w_out_val)
                tl.store(y_ptr + y_offset, output[:, h_idx, w_idx], 
                        mask=c_out_mask)


# Final optimized version - implementing the exact PyTorch transposed convolution behavior
@triton.jit
def transposed_conv2d_kernel_final(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C_in, H_in, W_in,
    C_out, K_h, K_w,
    H_out, W_out,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block computes one output element
    b_idx = tl.program_id(0)
    c_out_idx = tl.program_id(1)
    h_out_idx = tl.program_id(2)
    w_out_idx = tl.program_id(3)
    
    # Initialize accumulator
    acc = tl.zeros((1,), tl.float32)
    
    # Iterate over all input channels and kernel positions
    for c_in in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate corresponding input position
                # PyTorch transposed conv formula: h_out = (h_in - 1) * stride - 2*padding + (kh - 1) + output_padding + 1
                # So h_in = (h_out + 2*padding - (kh - 1) - output_padding) / stride + 1
                # But more intuitively, the input at (h_in, w_in) contributes to output at:
                # h_out = h_in * stride - padding + kh
                # => h_in = (h_out + padding - kh) / stride
                
                h_in = (h_out_idx + padding - kh) // stride
                w_in = (w_out_idx + padding - kw) // stride
                
                # Check if this is a valid contribution
                valid = ((h_out_idx + padding - kh) % stride == 0 and
                        (w_out_idx + padding - kw) % stride == 0 and
                        h_in >= 0 and h_in < H_in and
                        w_in >= 0 and w_in < W_in)
                
                if valid:
                    # Calculate pointers
                    x_offset = (b_idx * C_in * H_in * W_in +
                               c_in * H_in * W_in +
                               h_in * W_in + w_in)
                    w_offset = (c_in * C_out * K_h * K_w +
                               c_out_idx * K_h * K_w +
                               kh * K_w + kw)
                    
                    x_val = tl.load(x_ptr + x_offset)
                    w_val = tl.load(w_ptr + w_offset)
                    acc += x_val * w_val
    
    # Add bias
    if b_ptr is not None:
        b_val = tl.load(b_ptr + c_out_idx)
        acc += b_val
    
    # Store result
    y_offset = (b_idx * C_out * H_out * W_out +
               c_out_idx * H_out * W_out +
               h_out_idx * W_out + w_out_idx)
    tl.store(y_ptr + y_offset, acc)


def triton_transposed_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
):
    """
    Performs transposed 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (B, C_in, H_in, W_in)
        weight: Weight tensor of shape (C_in, C_out, K_h, K_w)
        bias: Optional bias tensor of shape (C_out,)
        stride, padding, output_padding, groups: Convolution parameters
        
    Returns:
        Output tensor of shape (B, C_out, H_out, W_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C_in, H_in, W_in = x.shape
    C_out, K_h, K_w = weight.shape[1], weight.shape[2], weight.shape[3]
    
    # Calculate output dimensions (same as PyTorch)
    H_out = (H_in - 1) * stride - 2 * padding + (K_h - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + (K_w - 1) + output_padding + 1
    
    # Prepare output tensor
    y = torch.empty(B, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Determine grid dimensions
    grid = (B, C_out, H_out, W_out)
    
    # Launch kernel
    transposed_conv2d_kernel_final[grid](
        x, weight, bias, y,
        B, C_in, H_in, W_in,
        C_out, K_h, K_w,
        H_out, W_out,
        stride, padding, output_padding,
        BLOCK_SIZE=1,
    )
    
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight and bias
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_transposed_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
        )