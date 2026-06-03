import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    x_ptr,  # Input: (B, C_in, D, H, W)
    w_ptr,  # Weight: (C_in, C_out, Kd, Kh, Kw)
    b_ptr,  # Bias: (C_out,) or None
    out_ptr,  # Output: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, C_out,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    # Output padding
    out_pad_d, out_pad_h, out_pad_w,
    # Block sizes
    BLOCK_C_in: tl.constexpr,
    BLOCK_D_out: tl.constexpr,
    BLOCK_H_out: tl.constexpr,
    BLOCK_W_out: tl.constexpr,
    BLOCK_Kd: tl.constexpr,
    BLOCK_Kh: tl.constexpr,
    BLOCK_Kw: tl.constexpr,
):
    # Program IDs for output dimensions
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d_out = tl.program_id(2)
    pid_h_out = tl.program_id(3)
    pid_w_out = tl.program_id(4)
    
    # Create ranges for output tiles
    d_offsets = pid_d_out * BLOCK_D_out + tl.arange(0, BLOCK_D_out)
    h_offsets = pid_h_out * BLOCK_H_out + tl.arange(0, BLOCK_H_out)
    w_offsets = pid_w_out * BLOCK_W_out + tl.arange(0, BLOCK_W_out)
    
    # Check bounds for output dimensions
    d_mask = d_offsets < D_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    
    # Calculate corresponding input positions for this output position
    # For transposed conv: input_pos = (output_pos - kernel_pos + stride - 1) // stride
    # But more intuitively: output_pos = input_pos * stride + kernel_pos - stride + 1 - pad
    
    # Accumulator for this output position
    output = tl.zeros((BLOCK_D_out, BLOCK_H_out, BLOCK_W_out), dtype=tl.float32)
    
    # Iterate over input channels in tiles
    for c_in_start in range(0, C_in, BLOCK_C_in):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_C_in)
        c_in_mask = c_in_offsets < C_in
        
        # Load input tile: shape (BLOCK_D_out, BLOCK_H_out, BLOCK_W_out, BLOCK_C_in)
        # We need to compute which input positions contribute to each output position
        # For each output position (d, h, w), it's affected by input positions where:
        # d_in = (d - k_d + stride_d - 1) // stride_d, but only if d = d_in * stride_d + k_d - stride_d + 1 - pad_d
        
        # Simplified: d_in * stride_d = d - k_d + stride_d - 1 + pad_d
        # So d_in = (d - k_d + stride_d - 1 + pad_d) / stride_d
        
        # For the kernel positions that contribute to each output position
        # k_d ranges from 0 to Kd-1
        # Valid input positions must be in [0, D_in-1]
        
        # Precompute the valid range of kernel positions for each output position
        # d_in = (d - k_d + stride_d - 1 + pad_d) / stride_d
        # 0 <= d_in < D_in
        # => 0 <= (d - k_d + stride_d - 1 + pad_d) / stride_d < D_in
        # => 0 <= d - k_d + stride_d - 1 + pad_d < D_in * stride_d
        # => k_d <= d + stride_d - 1 + pad_d and k_d > d + stride_d - 1 + pad_d - D_in * stride_d
        
        # For simplicity, iterate over all kernel positions and check validity
        
        # Process kernel in tiles
        for kd_start in range(0, Kd, BLOCK_Kd):
            kd_offsets = kd_start + tl.arange(0, BLOCK_Kd)
            kd_mask = kd_offsets < Kd
            
            for kh_start in range(0, Kh, BLOCK_Kh):
                kh_offsets = kh_start + tl.arange(0, BLOCK_Kh)
                kh_mask = kh_offsets < Kh
                
                for kw_start in range(0, Kw, BLOCK_Kw):
                    kw_offsets = kw_start + tl.arange(0, BLOCK_Kw)
                    kw_mask = kw_offsets < Kw
                    
                    # For each kernel position (kd, kh, kw), compute which output positions are affected
                    # And which input positions contribute
                    
                    # The relationship: output[d_out, h_out, w_out] += input[d_in, h_in, w_in] * kernel[c_in, c_out, kd, kh, kw]
                    # where d_out = d_in * stride_d + kd - stride_d + 1 - pad_d
                    # => d_in = (d_out - kd + stride_d - 1 + pad_d) / stride_d
                    
                    # Compute input positions for all output positions in our tile
                    d_in_vals = (d_offsets[:, None, None] - kd_offsets[None, :, None] + stride_d - 1 + pad_d) // stride_d
                    h_in_vals = (h_offsets[:, None, None] - kh_offsets[None, :, None] + stride_h - 1 + pad_h) // stride_h
                    w_in_vals = (w_offsets[None, :, None] - kw_offsets[None, None, :] + stride_w - 1 + pad_w) // stride_w
                    
                    # Check bounds for input positions
                    d_in_mask = (d_in_vals >= 0) & (d_in_vals < D_in)
                    h_in_mask = (h_in_vals >= 0) & (h_in_vals < H_in)
                    w_in_mask = (w_in_vals >= 0) & (w_in_vals < W_in)
                    valid_mask = d_in_mask & h_in_mask & w_in_mask
                    
                    # Compute flattened indices for input
                    d_in_flat = d_in_vals * (H_in * W_in)
                    h_in_flat = h_in_vals * (W_in)
                    w_in_flat = w_in_vals
                    input_indices = d_in_flat + h_in_flat + w_in_flat
                    
                    # Load input values (broadcasted)
                    # x_ptr shape: (B, C_in, D_in, H_in, W_in)
                    # We need to gather input values for each (d_out, h_out, w_out, c_in) combination
                    
                    # For simplicity, we'll compute in a more straightforward way:
                    # For each output position, sum over c_in and kernel positions
                    
                    # Reshape for easier indexing
                    # d_offsets: (BLOCK_D_out,)
                    # h_offsets: (BLOCK_H_out,)
                    # w_offsets: (BLOCK_W_out,)
                    
                    # Process each c_in in the tile
                    for c_in_offset in range(BLOCK_C_in):
                        c_in_idx = c_in_start + c_in_offset
                        if c_in_idx >= C_in:
                            break
                            
                        # Calculate input pointer offset for this c_in
                        x_offset = pid_b * (C_in * D_in * H_in * W_in) + c_in_idx * (D_in * H_in * W_in)
                        
                        # Process each kernel position
                        for kd_idx in range(BLOCK_Kd):
                            kd = kd_start + kd_idx
                            if kd >= Kd:
                                break
                                
                            for kh_idx in range(BLOCK_Kh):
                                kh = kh_start + kh_idx
                                if kh >= Kh:
                                    break
                                    
                                for kw_idx in range(BLOCK_Kw):
                                    kw = kw_start + kw_idx
                                    if kw >= Kw:
                                        break
                                    
                                    # Calculate corresponding output positions that this kernel position affects
                                    # d_out = d_in * stride_d + kd - stride_d + 1 - pad_d
                                    d_out_min = kd - stride_d + 1 - pad_d
                                    h_out_min = kh - stride_h + 1 - pad_h
                                    w_out_min = kw - stride_w + 1 - pad_w
                                    
                                    # Only process if there's overlap with our output tile
                                    d_out_start = max(0, (d_out_min + stride_d - 1) // stride_d)
                                    h_out_start = max(0, (h_out_min + stride_h - 1) // stride_h)
                                    w_out_start = max(0, (w_out_min + stride_w - 1) // stride_w)
                                    
                                    d_out_end = min(D_out, (d_out_min + D_in * stride_d + stride_d - 1) // stride_d)
                                    h_out_end = min(H_out, (h_out_min + H_in * stride_h + stride_h - 1) // stride_h)
                                    w_out_end = min(W_out, (w_out_min + W_in * stride_w + stride_w - 1) // stride_w)
                                    
                                    if d_out_start >= d_out_end or h_out_start >= h_out_end or w_out_start >= w_out_end:
                                        continue
                                    
                                    # Calculate input indices for the valid range
                                    d_in_start = (d_out_start * stride_d - kd + stride_d - 1 + pad_d) // stride_d
                                    h_in_start = (h_out_start * stride_h - kh + stride_h - 1 + pad_h) // stride_h
                                    w_in_start = (w_out_start * stride_w - kw + stride_w - 1 + pad_w) // stride_w
                                    
                                    # Load input values for this c_in
                                    x_ptr_cin = x_ptr + x_offset
                                    
                                    # Calculate weight index
                                    w_offset = c_in_idx * (C_out * Kd * Kh * Kw) + pid_c_out * (Kd * Kh * Kw) + kd * (Kh * Kw) + kh * Kw + kw
                                    w_val = tl.load(w_ptr + w_offset)
                                    
                                    # Accumulate contributions
                                    for d_out_idx in range(d_out_start, d_out_end):
                                        d_in = d_in_start + (d_out_idx - d_out_start)
                                        if d_in < 0 or d_in >= D_in:
                                            continue
                                            
                                        d_in_offset = d_in * (H_in * W_in)
                                        
                                        for h_out_idx in range(h_out_start, h_out_end):
                                            h_in = h_in_start + (h_out_idx - h_out_start)
                                            if h_in < 0 or h_in >= H_in:
                                                continue
                                                
                                            h_in_offset = h_in * W_in
                                            
                                            for w_out_idx in range(w_out_start, w_out_end):
                                                w_in = w_in_start + (w_out_idx - w_out_start)
                                                if w_in < 0 or w_in >= W_in:
                                                    continue
                                                    
                                                # Calculate input value index
                                                x_idx = d_in_offset + h_in_offset + w_in
                                                x_val = tl.load(x_ptr_cin + x_idx)
                                                
                                                # Accumulate to output
                                                output[d_out_idx - pid_d_out * BLOCK_D_out, 
                                                       h_out_idx - pid_h_out * BLOCK_H_out, 
                                                       w_out_idx - pid_w_out * BLOCK_W_out] += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + pid_c_out)
        output += b_val
    
    # Store output
    # Check bounds for output positions
    d_mask_3d = d_mask[:, None, None]
    h_mask_3d = h_mask[None, :, None]
    w_mask_3d = w_mask[None, None, :]
    combined_mask = d_mask_3d & h_mask_3d & w_mask_3d
    
    out_offset = (pid_b * (C_out * D_out * H_out * W_out) + 
                  pid_c_out * (D_out * H_out * W_out) +
                  d_offsets[:, None, None] * (H_out * W_out) +
                  h_offsets[None, :, None] * W_out +
                  w_offsets[None, None, :])
    
    tl.store(out_ptr + out_offset, output, mask=combined_mask)


class ConvTranspose3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, groups):
        # Save context for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        
        # Get output shape
        B, C_in, D_in, H_in, W_in = x.shape
        C_out = weight.shape[1]  # Weight shape: (C_in, C_out, Kd, Kh, Kw)
        Kd, Kh, Kw = weight.shape[2], weight.shape[3], weight.shape[4]
        stride_d, stride_h, stride_w = stride
        pad_d, pad_h, pad_w = padding
        out_pad_d, out_pad_h, out_pad_w = output_padding
        
        # Calculate output dimensions
        D_out = (D_in - 1) * stride_d - 2 * pad_d + Kd + out_pad_d
        H_out = (H_in - 1) * stride_h - 2 * pad_h + Kh + out_pad_h
        W_out = (W_in - 1) * stride_w - 2 * pad_w + Kw + out_pad_w
        
        # Allocate output
        out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Grid dimensions
        grid = lambda meta: (
            B,  # batch size
            C_out,  # output channels
            triton.cdiv(D_out, meta['BLOCK_D_out']),
            triton.cdiv(H_out, meta['BLOCK_H_out']),
            triton.cdiv(W_out, meta['BLOCK_W_out']),
        )
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x, weight, bias, out,
            B, C_in, C_out,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            Kd, Kh, Kw,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            out_pad_d, out_pad_h, out_pad_w,
            BLOCK_C_in=8,
            BLOCK_D_out=4,
            BLOCK_H_out=4,
            BLOCK_W_out=4,
            BLOCK_Kd=3,
            BLOCK_Kh=3,
            BLOCK_Kw=3,
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch implementation for backward
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        groups = ctx.groups
        
        # Use PyTorch's built-in backward
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.ops.aten.convolution_backward(
                grad_output, x, weight, None, stride, padding, 
                output_padding, groups, False, False, False, 
                [True, False, False])[0]
        
        return grad_input, grad_weight, grad_bias, None, None, None, None


def triton_conv_transpose3d(x, weight, bias=None, stride=(1, 1, 1), 
                           padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1):
    return ConvTranspose3dFunction.apply(x, weight, bias, stride, padding, output_padding, groups)


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weights
        Kd, Kh, Kw = kernel_size
        weight = torch.empty(in_channels, out_channels, Kd, Kh, Kw)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)
        
        # Initialize bias if needed
        if bias:
            bound = 1 / math.sqrt(in_channels * Kd * Kh * Kw)
            bias_tensor = torch.empty(out_channels).uniform_(-bound, bound)
            self.bias = nn.Parameter(bias_tensor)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(x, self.weight, self.bias, 
                                      self.stride, self.padding, 
                                      self.output_padding, self.groups)