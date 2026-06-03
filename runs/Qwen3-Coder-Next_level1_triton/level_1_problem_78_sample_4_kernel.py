import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def transposed_conv2d_kernel(
    # Pointers to tensors
    x_ptr,  # Input: (B, C_in, H_in, W_in)
    w_ptr,  # Weight: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias: (C_out,) or None
    out_ptr,  # Output: (B, C_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, H_in, W_in,
    C_out, K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    # Output dimensions
    H_out, W_out,
    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_c_in, w_stride_c_out, w_stride_kh, w_kw,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    # Meta-parameters
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute output position
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Check bounds for output
    out_h_mask = out_h < H_out
    out_w_mask = out_w < W_out
    out_h = tl.maximum(tl.minimum(out_h, H_out - 1), 0)
    out_w = tl.maximum(tl.minimum(out_w, W_out - 1), 0)
    
    # Compute corresponding input positions for each output position
    # For transposed convolution: out_h = in_h * stride_h + kernel_offset - pad_h
    # So in_h = (out_h + pad_h - kernel_offset) // stride_h
    in_h_start = (out_h[:, None] + pad_h) // stride_h
    in_w_start = (out_w[None, :] + pad_w) // stride_w
    
    # Kernel offsets (0 to K_h-1, 0 to K_w-1)
    kh = tl.arange(0, BLOCK_SIZE_K)
    kw = tl.arange(0, BLOCK_SIZE_K)
    
    # Compute input positions for each kernel position
    in_h = in_h_start[:, None, :] - (kh[None, :, None] * stride_h - pad_h) // stride_h + (kh[None, :, None] * stride_h - pad_h) % stride_h // stride_h
    in_w = in_w_start[None, :, :] - (kw[None, None, :] * stride_w - pad_w) // stride_w + (kw[None, None, :] * stride_w - pad_w) % stride_w // stride_w
    
    # Simplify: For standard transposed conv with stride s, kernel k:
    # out_pos = in_pos * stride + kernel_pos - pad
    # So in_pos = (out_pos + pad - kernel_pos) / stride
    # Only use this in_pos if (out_pos + pad - kernel_pos) is divisible by stride
    
    # Better approach: iterate over input channels and kernel positions
    # For each output position, accumulate: sum_{c_in, kh, kw} x[b, c_in, in_h, in_w] * w[c_in, c_out, kh, kw]
    # where in_h = (out_h + pad_h - kh) // stride_h, in_w = (out_w + pad_w - kw) // stride_w
    
    # Accumulator for the output
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels in blocks
    for c_in_offset in range(0, C_in, BLOCK_SIZE_C_IN):
        c_in_ids = c_in_offset + tl.arange(0, BLOCK_SIZE_C_IN)
        c_in_mask = c_in_ids < C_in
        
        # Load weights for this block: shape (BLOCK_SIZE_C_IN, BLOCK_SIZE_C_OUT, K_h, K_w)
        # But we need to iterate over kernel positions
        for kh_offset in range(0, K_h, BLOCK_SIZE_K):
            for kw_offset in range(0, K_w, BLOCK_SIZE_K):
                kh_ids = kh_offset + tl.arange(0, BLOCK_SIZE_K)
                kw_ids = kw_offset + tl.arange(0, BLOCK_SIZE_K)
                
                kh_mask = kh_ids < K_h
                kw_mask = kw_ids < K_w
                
                # Compute input positions for this kernel position
                # in_h = (out_h + pad_h - kh) // stride_h
                in_h_vals = (out_h[:, None] + pad_h - kh_ids[None, :]) // stride_h
                in_w_vals = (out_w[None, :] + pad_w - kw_ids[None, :]) // stride_w
                
                # Check if input positions are valid
                h_valid = (in_h_vals >= 0) & (in_h_vals < H_in)
                w_valid = (in_w_vals >= 0) & (in_w_vals < W_in)
                valid_mask = h_valid[:, :, None] & w_valid[:, :, None] & c_in_mask[None, None, :]
                
                # Load input values where valid
                x_ptrs = x_ptr + pid_b * x_stride_b + c_in_ids[None, None, :] * x_stride_c + \
                         in_h_vals[:, :, None] * x_stride_h + in_w_vals[:, :, None] * x_stride_w
                
                # Create mask for loading
                load_mask = valid_mask
                
                x_vals = tl.load(x_ptrs, mask=load_mask, other=0.0)
                
                # Load weights: w[c_in, c_out, kh, kw]
                w_ptrs = w_ptr + c_in_ids[:, None, None] * w_stride_c_in + \
                         pid_c_out * w_stride_c_out + \
                         kh_ids[None, :, None] * w_stride_kh + kw_ids[None, None, :] * w_kw
                
                # Load weights with proper mask
                w_mask = kh_mask[None, :, None] & kw_mask[None, None, :] & c_in_mask[:, None, None]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
                
                # Compute contribution: x * w for all combinations
                # x_vals: (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C_IN)
                # w_vals: (BLOCK_SIZE_C_IN, BLOCK_SIZE_K, BLOCK_SIZE_K)
                # We need to align dimensions properly
                
                # Reshape for broadcasting
                x_vals_expanded = x_vals[:, :, :, None, None]  # (H, W, C_in, 1, 1)
                w_vals_expanded = w_vals[None, None, :, :, :]  # (1, 1, C_in, K_h, K_w)
                
                # But we only want the valid kernel positions
                # Actually, let's restructure: for each output position (h,w), 
                # accumulate over c_in, kh, kw
                
                # Better: compute outer product and sum
                # acc[h,w] += sum_{c_in, kh, kw} x[b,c_in,in_h,in_w] * w[c_in,c_out,kh,kw]
                
                # Simplified approach: for each valid (h,w) position and kernel offset
                for i in range(BLOCK_SIZE_K):
                    for j in range(BLOCK_SIZE_K):
                        kh_idx = kh_offset + i
                        kw_idx = kw_offset + j
                        
                        if kh_idx < K_h and kw_idx < K_w:
                            # Compute input positions for this kernel position
                            in_h_idx = (out_h[:, None] + pad_h - kh_idx) // stride_h
                            in_w_idx = (out_w[None, :] + pad_w - kw_idx) // stride_w
                            
                            h_valid_i = (in_h_idx >= 0) & (in_h_idx < H_in)
                            w_valid_i = (in_w_idx >= 0) & (in_w_idx < W_in)
                            valid_i = h_valid_i & w_valid_i
                            
                            # Load input at this position
                            x_ptrs_i = x_ptr + pid_b * x_stride_b + c_in_ids[None, None] * x_stride_c + \
                                      in_h_idx * x_stride_h + in_w_idx * x_stride_w
                            x_vals_i = tl.load(x_ptrs_i, mask=valid_i[:, :, None], other=0.0)
                            
                            # Load weight at this kernel position
                            w_ptrs_i = w_ptr + c_in_ids[:, None] * w_stride_c_in + \
                                      pid_c_out * w_stride_c_out + \
                                      kh_idx * w_stride_kh + kw_idx * w_kw
                            w_vals_i = tl.load(w_ptrs_i, mask=c_in_mask[:, None], other=0.0)
                            
                            # Accumulate: x_vals_i has shape (H_out_block, W_out_block, C_in_block)
                            # w_vals_i has shape (C_in_block, 1)
                            # Result: (H_out_block, W_out_block)
                            contrib = tl.sum(x_vals_i * w_vals_i[None, None, :], axis=2)
                            acc += tl.where(valid_i[:, :, None], contrib, 0.0).sum(axis=2)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Store result
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_c_out * out_stride_c + \
               out_h[:, None] * out_stride_h + out_w[None, :] * out_stride_w
    out_mask = (out_h[:, None] < H_out) & (out_w[None, :] < W_out)
    tl.store(out_ptrs, acc, mask=out_mask)


# A more practical implementation using a different tiling strategy
@triton.jit
def transposed_conv2d_kernel_v2(
    x_ptr,  # Input: (B, C_in, H_in, W_in)
    w_ptr,  # Weight: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias: (C_out,) or None
    out_ptr,  # Output: (B, C_out, H_out, W_out)
    # Dimensions
    B, C_in, H_in, W_in,
    C_out, K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    H_out, W_out,
    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_c_in, w_stride_c_out, w_stride_kh, w_kw,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    # Block sizes
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get output position
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Output indices
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Masks
    out_h_mask = out_h < H_out
    out_w_mask = out_w < W_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in in range(C_in):
        # For each kernel position
        for kh in range(K_h):
            for kw in range(K_w):
                # Compute input position that contributes to this output
                in_h = (out_h[:, None] + pad_h - kh) // stride_h
                in_w = (out_w[None, :] + pad_w - kw) // stride_w
                
                # Check validity
                h_valid = (in_h >= 0) & (in_h < H_in)
                w_valid = (in_w >= 0) & (in_w < W_in)
                valid = h_valid & w_valid
                
                # Load input value
                x_h = tl.maximum(tl.minimum(in_h, H_in - 1), 0)
                x_w = tl.maximum(tl.minimum(in_w, W_in - 1), 0)
                x_val = tl.load(x_ptr + pid_b * x_stride_b + c_in * x_stride_c + 
                               x_h * x_stride_h + x_w * x_stride_w, 
                               mask=valid, other=0.0)
                
                # Load weight
                w_val = tl.load(w_ptr + c_in * w_stride_c_in + pid_c_out * w_stride_c_out + 
                               kh * w_stride_kh + kw * w_kw)
                
                # Accumulate
                acc += tl.where(valid, x_val * w_val, 0.0)
    
    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Store output
    out_h_clamped = tl.maximum(tl.minimum(out_h, H_out - 1), 0)
    out_w_clamped = tl.maximum(tl.minimum(out_w, W_out - 1), 0)
    tl.store(out_ptr + pid_b * out_stride_b + pid_c_out * out_stride_c + 
            out_h_clamped[:, None] * out_stride_h + out_w_clamped[None, :] * out_stride_w,
            acc, mask=out_h_mask[:, None] & out_w_mask[None, :])


def triton_transposed_conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of 2D transposed convolution.
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    B, C_in, H_in, W_in = x.shape
    C_in_w, C_out, K_h, K_w = weight.shape
    
    assert C_in == C_in_w, f"Input channels {C_in} must match weight input channels {C_in_w}"
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h + output_padding[0]
    W_out = (W_in - 1) * stride_w - 2 * pad_w + K_w + output_padding[1]
    
    # Allocate output tensor
    out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Compute strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
    w_stride_c_in, w_stride_c_out, w_stride_kh, w_kw = weight.stride()
    out_stride_b, out_stride_c, out_stride_h, out_stride_w = out.stride()
    
    # Kernel block sizes (tunable)
    BLOCK_SIZE_C_OUT = 16
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    # Grid dimensions
    grid = (B, triton.cdiv(C_out, BLOCK_SIZE_C_OUT), 
            triton.cdiv(H_out, BLOCK_SIZE_H), triton.cdiv(W_out, BLOCK_SIZE_W))
    
    # Launch kernel
    transposed_conv2d_kernel_v2[grid](
        x, weight, bias, out,
        B, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        H_out, W_out,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        w_stride_c_in, w_stride_c_out, w_stride_kh, w_kw,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed convolution model using custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the same convolution layer but we'll override the forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        
        # Initialize weights and bias (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights using the same method as ConvTranspose2d
        # Fan-in calculation for transposed conv
        receptive_size = kernel_size[0] * kernel_size[1]
        fan_in = in_channels * receptive_size
        bound = 1 / math.sqrt(fan_in)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if bias:
                self.bias.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using custom Triton kernel.
        """
        return triton_transposed_conv2d(
            x, self.weight, self.bias, 
            stride=self.stride, padding=self.padding
        )