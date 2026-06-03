import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to tensors
    x_ptr,  # Input: (B, C_in, H_in, W_in)
    w_ptr,  # Weight: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias: (C_out,) or None
    y_ptr,  # Output: (B, C_out, H_out, W_out)
    # Dimensions
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    K_h: tl.constexpr,
    K_w: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    # Block sizes for tiling
    BLOCK_B: tl.constexpr,
    BLOCK_C_out: tl.constexpr,
    BLOCK_H_out: tl.constexpr,
    BLOCK_W_out: tl.constexpr,
    BLOCK_K: tl.constexpr,  # for grouping input/output channels
):
    # Get program IDs
    batch_idx = tl.program_id(0) // (C_out // BLOCK_C_out)
    c_out_block = tl.program_id(0) % (C_out // BLOCK_C_out)
    
    h_block = tl.program_id(1)
    w_block = tl.program_id(2)
    
    # Compute actual indices
    batch_idx = batch_idx % B
    c_out_start = c_out_block * BLOCK_C_out
    h_start = h_block * BLOCK_H_out
    w_start = w_block * BLOCK_W_out
    
    # Create ranges for output dimensions
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_C_out)
    h_offsets = h_start + tl.arange(0, BLOCK_H_out)
    w_offsets = w_start + tl.arange(0, BLOCK_W_out)
    
    # Create mask for output bounds
    c_out_mask = c_out_offsets < C_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_C_out, BLOCK_H_out, BLOCK_W_out), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for c_in_block in range(0, C_in, BLOCK_K):
        c_in_offsets = c_in_block + tl.arange(0, BLOCK_K)
        c_in_mask = c_in_offsets < C_in
        
        # Load input: [BLOCK_K, BLOCK_H_out, BLOCK_W_out]
        # We need to compute which input positions contribute to this output
        # For transposed conv: H_in = (H_out - 1 - 2*padding - output_padding + K_h) // stride + 1
        # So input H index = (output_H - stride) // stride + kernel_H_pos (with offsets)
        
        # Compute input H indices for this output tile
        for i_h in range(BLOCK_H_out):
            for i_w in range(BLOCK_W_out):
                out_h = h_start + i_h
                out_w = w_start + i_w
                
                # For transposed convolution, the relationship is:
                # out_h = in_h * stride + (kernel_h - 1 - padding) - extra_padding
                # So in_h = (out_h - (kernel_h - 1 - padding) + extra_padding) / stride
                # Simplified: in_h = (out_h - padding + extra_padding_offset) // stride
                
                # Calculate the range of kernel positions that contribute
                # and corresponding input positions
                
                # For transposed conv with stride S:
                # output[H_out] gets contributions from input[H_in] where
                # H_out = H_in * stride + (K_h - 1 - pad) + extra_pad
                
                # So H_in = (H_out - (K_h - 1 - pad) - extra_pad) / stride
                
                # Let's compute the input position for this output position
                # and kernel position
                for kh in range(K_h):
                    for kw in range(K_w):
                        # Calculate input position that would contribute
                        # For transposed conv: out_pos = in_pos * stride + (kernel_pos - padding) + output_padding_offset
                        # Actually, the standard formula is:
                        # out_h = in_h * stride - padding + kh
                        # => in_h = (out_h + padding - kh) // stride
                        
                        in_h = (out_h + padding - kh) // stride
                        in_w = (out_w + padding - kw) // stride
                        
                        # Check if this input position is valid
                        if in_h >= 0 and in_h < H_in and in_w >= 0 and in_w < W_in:
                            # Check if this kernel position is valid for this output
                            if (out_h + padding - kh) % stride == 0 and (out_w + padding - kw) % stride == 0:
                                # Load input value
                                in_h_offset = in_h
                                in_w_offset = in_w
                                
                                # Get input pointer offset: B*C_in*H_in*W_in batch offset + ...
                                input_offset = (
                                    batch_idx * C_in * H_in * W_in +
                                    c_in_offsets[:, None, None] * H_in * W_in +
                                    in_h_offset * W_in +
                                    in_w_offset
                                )
                                
                                # Reshape for broadcasting: (BLOCK_K, 1, 1)
                                input_val = tl.load(
                                    x_ptr + input_offset,
                                    mask=c_in_mask[:, None, None] & 
                                         (in_h_offset < H_in) & 
                                         (in_w_offset < W_in),
                                    other=0.0
                                )
                                
                                # Load weight: w[c_in, c_out, kh, kw]
                                weight_offset = (
                                    c_in_offsets[:, None, None] * C_out * K_h * K_w +
                                    c_out_offsets[None, :, None] * K_h * K_w +
                                    kh * K_w +
                                    kw
                                )
                                
                                weight_val = tl.load(
                                    w_ptr + weight_offset,
                                    mask=c_in_mask[:, None, None] & c_out_mask[None, :, None],
                                    other=0.0
                                )
                                
                                # Accumulate: input * weight
                                acc += input_val * weight_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_offsets, mask=c_out_mask)
        acc += bias[None, :, None]  # Add bias over channel dimension
    
    # Store result
    y_offset = (
        batch_idx * C_out * H_out * W_out +
        c_out_offsets[:, None, None] * H_out * W_out +
        h_offsets[None, :, None] * W_out +
        w_offsets[None, None, :]
    )
    
    tl.store(
        y_ptr + y_offset,
        acc,
        mask=c_out_mask[:, None, None] & 
             h_mask[None, :, None] & 
             w_mask[None, None, :]
    )


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride=1, padding=0, output_padding=0):
        # Save context for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        
        # Compute output shape manually
        B, C_in, H_in, W_in = x.shape
        C_out, _, K_h, K_w = weight.shape
        
        H_out = (H_in - 1) * stride - 2 * padding + output_padding + K_h
        W_out = (W_in - 1) * stride - 2 * padding + output_padding + K_w
        
        # Create output tensor
        y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
        
        # Define block sizes for tiling
        BLOCK_B = 1
        BLOCK_C_out = 16
        BLOCK_H_out = 32
        BLOCK_W_out = 32
        BLOCK_K = 16
        
        # Calculate grid dimensions
        grid = lambda meta: (
            B * (C_out + meta["BLOCK_C_out"] - 1) // meta["BLOCK_C_out"],
            (H_out + meta["BLOCK_H_out"] - 1) // meta["BLOCK_H_out"],
            (W_out + meta["BLOCK_W_out"] - 1) // meta["BLOCK_W_out"]
        )
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias, y,
            B, C_in, C_out, K_h, K_w,
            H_in, W_in, H_out, W_out,
            stride, padding, output_padding,
            BLOCK_B=BLOCK_B,
            BLOCK_C_out=BLOCK_C_out,
            BLOCK_H_out=BLOCK_H_out,
            BLOCK_W_out=BLOCK_W_out,
            BLOCK_K=BLOCK_K
        )
        
        return y


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0):
    """Wrapper function for the Triton transposed convolution"""
    return TritonConvTranspose2d.apply(x, weight, bias, stride, padding, output_padding)


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights using kaiming_uniform as in PyTorch"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding
        )


import math