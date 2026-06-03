import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to tensors
    x_ptr,  # Input: (B, C_in, H_in, W_in)
    w_ptr,  # Weight: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias: (C_out,) or None
    y_ptr,  # Output: (B, C_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, H_in, W_in,
    C_out, K_h, K_w,
    H_out, W_out,
    # Stride and padding parameters
    stride_h, stride_w,
    pad_h, pad_w,
    out_pad_h, out_pad_w,
    # Block sizes for tiling
    BLOCK_SIZE_C_in: tl.constexpr,
    BLOCK_SIZE_C_out: tl.constexpr,
    BLOCK_SIZE_K_h: tl.constexpr,
    BLOCK_SIZE_K_w: tl.constexpr,
    BLOCK_SIZE_H_out: tl.constexpr,
    BLOCK_SIZE_W_out: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    c_out_block = tl.program_id(1)
    h_out_block = tl.program_id(2)
    w_out_block = tl.program_id(3)
    
    # Calculate output feature channel range
    c_out_start = c_out_block * BLOCK_SIZE_C_out
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_out)
    c_out_mask = c_out_offsets < C_out
    
    # Calculate output height range
    h_out_start = h_out_block * BLOCK_SIZE_H_out
    h_out_offsets = h_out_start + tl.arange(0, BLOCK_SIZE_H_out)
    h_out_mask = h_out_offsets < H_out
    
    # Calculate output width range
    w_out_start = w_out_block * BLOCK_SIZE_W_out
    w_out_offsets = w_out_start + tl.arange(0, BLOCK_SIZE_W_out)
    w_out_mask = w_out_offsets < W_out
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_C_out, BLOCK_SIZE_H_out, BLOCK_SIZE_W_out), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_block in range(0, C_in, BLOCK_SIZE_C_in):
        c_in_start = c_in_block
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C_in)
        c_in_mask = c_in_offsets < C_in
        
        # Loop over kernel height
        for k_h_block in range(0, K_h, BLOCK_SIZE_K_h):
            k_h_start = k_h_block
            k_h_offsets = k_h_start + tl.arange(0, BLOCK_SIZE_K_h)
            k_h_mask = k_h_offsets < K_h
            
            # Loop over kernel width
            for k_w_block in range(0, K_w, BLOCK_SIZE_K_w):
                k_w_start = k_w_block
                k_w_offsets = k_w_start + tl.arange(0, BLOCK_SIZE_K_w)
                k_w_mask = k_w_offsets < K_w
                
                # Calculate corresponding input positions
                # For transposed convolution: h_in = (h_out - out_pad_h - 1 + k_h) // stride_h
                h_in_offsets = (h_out_offsets[None, :, None] - out_pad_h - 1 + k_h_offsets[:, None, None]) // stride_h
                w_in_offsets = (w_out_offsets[None, None, :] - out_pad_w - 1 + k_w_offsets[None, None, :]) // stride_w
                
                # Check if input positions are valid
                h_in_valid = (h_in_offsets >= 0) & (h_in_offsets < H_in)
                w_in_valid = (w_in_offsets >= 0) & (w_in_offsets < W_in)
                valid_mask = h_in_valid & w_in_valid
                
                # Load input values: shape (BLOCK_SIZE_C_in, BLOCK_SIZE_H_out, BLOCK_SIZE_W_out)
                x_offsets = (
                    batch_id * (C_in * H_in * W_in) +
                    c_in_offsets[:, None, None] * (H_in * W_in) +
                    h_in_offsets[None, :, :] * W_in +
                    w_in_offsets[None, :, :]
                )
                x_vals = tl.load(
                    x_ptr + x_offsets,
                    mask=c_in_mask[:, None, None] & valid_mask[None, :, :],
                    other=0.0
                )
                
                # Load weight values: shape (BLOCK_SIZE_C_in, BLOCK_SIZE_C_out, BLOCK_SIZE_K_h, BLOCK_SIZE_K_w)
                w_offsets = (
                    c_in_offsets[:, None, None, None] * (C_out * K_h * K_w) +
                    c_out_offsets[None, :, None, None] * (K_h * K_w) +
                    k_h_offsets[None, None, :, None] * K_w +
                    k_w_offsets[None, None, None, :]
                )
                w_vals = tl.load(
                    w_ptr + w_offsets,
                    mask=c_in_mask[:, None, None, None] & c_out_mask[None, :, None, None] & 
                         k_h_mask[None, None, :, None] & k_w_mask[None, None, None, :],
                    other=0.0
                )
                
                # Compute contribution to output: x * w
                # x_vals: (C_in_block, H_out_block, W_out_block)
                # w_vals: (C_in_block, C_out_block, K_h_block, K_w_block)
                # We need to align dimensions for multiplication
                
                # Reshape for broadcasting: x_vals -> (C_in_block, 1, H_out_block, W_out_block)
                # w_vals -> (C_in_block, C_out_block, K_h_block, K_w_block)
                # Result should accumulate to (C_out_block, H_out_block, W_out_block)
                
                # Use Einstein notation or explicit summation
                # For each input channel, multiply with corresponding kernel weights
                # and accumulate to output channels
                
                # Expand x_vals for broadcasting: (C_in_block, 1, H_out_block, W_out_block)
                x_expanded = x_vals[:, None, :, :]
                # Multiply and sum over input channels
                # w_vals: (C_in_block, C_out_block, K_h_block, K_w_block)
                # x_expanded: (C_in_block, 1, H_out_block, W_out_block)
                # We need to sum over C_in_block and the kernel spatial dimensions
                
                # Reshape to enable tensor contraction
                # Flatten kernel spatial dims: (C_in_block, C_out_block, K_h_block * K_w_block)
                w_reshaped = tl.reshape(w_vals, (BLOCK_SIZE_C_in, BLOCK_SIZE_C_out, BLOCK_SIZE_K_h * BLOCK_SIZE_K_w))
                # Flatten x spatial dims: (C_in_block, BLOCK_SIZE_H_out * BLOCK_SIZE_W_out)
                x_reshaped = tl.reshape(x_expanded, (BLOCK_SIZE_C_in, BLOCK_SIZE_H_out * BLOCK_SIZE_W_out))
                
                # Matrix multiplication: (C_in_block, C_out_block, K_h_block * K_w_block) @ (C_in_block, H_out_block * W_out_block)
                # But we need to handle the valid mask for each kernel position
                
                # Instead, let's do explicit accumulation with masking
                for k_h_idx in range(BLOCK_SIZE_K_h):
                    for k_w_idx in range(BLOCK_SIZE_K_w):
                        if k_h_offsets[k_h_idx] < K_h and k_w_offsets[k_w_idx] < K_w:
                            # Get the valid mask for this kernel position
                            kernel_valid = valid_mask & (k_h_offsets[k_h_idx] < K_h) & (k_w_offsets[k_w_idx] < K_w)
                            
                            # Load weight for this kernel position: shape (C_in_block, C_out_block)
                            w_kernel = w_vals[:, :, k_h_idx, k_w_idx]  # (BLOCK_SIZE_C_in, BLOCK_SIZE_C_out)
                            
                            # Load input: shape (C_in_block, H_out_block, W_out_block)
                            x_val = x_vals[:, :, :]  # (BLOCK_SIZE_C_in, BLOCK_SIZE_H_out, BLOCK_SIZE_W_out)
                            
                            # Multiply and accumulate
                            # x_val[:, None, :, :] * w_kernel[None, :, None, None] -> (C_in_block, C_out_block, H_out_block, W_out_block)
                            # Sum over C_in_block to get (C_out_block, H_out_block, W_out_block)
                            
                            # Reshape for efficient computation
                            x_flat = tl.reshape(x_val, (BLOCK_SIZE_C_in, BLOCK_SIZE_H_out * BLOCK_SIZE_W_out))
                            w_flat = tl.reshape(w_kernel, (BLOCK_SIZE_C_in, BLOCK_SIZE_C_out))
                            
                            # Matrix multiply: (C_in_block, H_out_block*W_out_block)^T @ (C_in_block, C_out_block)
                            # = (H_out_block*W_out_block, C_in_block) @ (C_in_block, C_out_block)
                            # = (H_out_block*W_out_block, C_out_block)
                            contrib = tl.dot(x_flat.T, w_flat, out_dtype=tl.float32)
                            contrib = tl.reshape(contrib, (BLOCK_SIZE_H_out, BLOCK_SIZE_W_out, BLOCK_SIZE_C_out))
                            contrib = tl.permute(contrib, (2, 0, 1))  # (C_out_block, H_out_block, W_out_block)
                            
                            acc += contrib
    
    # Store result
    y_offsets = (
        batch_id * (C_out * H_out * W_out) +
        c_out_offsets[:, None, None] * (H_out * W_out) +
        h_out_offsets[None, :, None] * W_out +
        w_out_offsets[None, None, :]
    )
    
    # Add bias if available
    if b_ptr is not None:
        bias_offsets = c_out_offsets
        bias_vals = tl.load(b_ptr + bias_offsets, mask=c_out_mask)
        acc += bias_vals[:, None, None]
    
    # Store the result
    tl.store(y_ptr + y_offsets, acc, mask=c_out_mask[:, None, None] & h_out_mask[None, :, None] & w_out_mask[None, None, :])


def triton_conv_transpose2d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose2d.
    
    Args:
        x: Input tensor of shape (B, C_in, H_in, W_in)
        weight: Weight tensor of shape (C_in, C_out, K_h, K_w)
        bias: Bias tensor of shape (C_out,) or None
        stride: stride height and width
        padding: padding height and width
        output_padding: additional size added to one side of output
        groups: number of groups for convolution (should be 1 for full convolution)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, H_in, W_in = x.shape
    C_in_w, C_out, K_h, K_w = weight.shape
    
    assert C_in == C_in_w, f"Input channels {C_in} doesn't match weight input channels {C_in_w}"
    assert groups == 1, "Only groups=1 is supported in this implementation"
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride - 2 * padding + K_h + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + K_w + output_padding
    
    # Prepare output tensor
    y = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_C_in = 16
    BLOCK_SIZE_C_out = 16
    BLOCK_SIZE_K_h = 3
    BLOCK_SIZE_K_w = 7
    BLOCK_SIZE_H_out = 8
    BLOCK_SIZE_W_out = 8
    
    # Grid dimensions: (batch, C_out_blocks, H_out_blocks, W_out_blocks)
    grid = lambda meta: (
        B,
        (C_out + meta['BLOCK_SIZE_C_out'] - 1) // meta['BLOCK_SIZE_C_out'],
        (H_out + meta['BLOCK_SIZE_H_out'] - 1) // meta['BLOCK_SIZE_H_out'],
        (W_out + meta['BLOCK_SIZE_W_out'] - 1) // meta['BLOCK_SIZE_W_out']
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, y,
        B, C_in, H_in, W_in,
        C_out, K_h, K_w,
        H_out, W_out,
        stride, stride,
        padding, padding,
        output_padding, output_padding,
        BLOCK_SIZE_C_in=BLOCK_SIZE_C_in,
        BLOCK_SIZE_C_out=BLOCK_SIZE_C_out,
        BLOCK_SIZE_K_h=BLOCK_SIZE_K_h,
        BLOCK_SIZE_K_w=BLOCK_SIZE_K_w,
        BLOCK_SIZE_H_out=BLOCK_SIZE_H_out,
        BLOCK_SIZE_W_out=BLOCK_SIZE_W_out,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
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
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Kaiming uniform initialization for transposed conv
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using the custom Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding,
            groups=self.groups
        )