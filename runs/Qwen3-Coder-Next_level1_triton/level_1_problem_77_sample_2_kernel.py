import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, D, H, W)
    w_ptr,  # Weight tensor pointer (C_in, C_out, K_d, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (N, C_out, D_out, H_out, W_out)
    N, C_in, C_out,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    K_d, K_h, K_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr = 8,
    BLOCK_SIZE_C_out: tl.constexpr = 32,
    BLOCK_SIZE_C_in: tl.constexpr = 8,
    BLOCK_SIZE_K: tl.constexpr = 4,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    c_out_block = tl.program_id(1)
    d_out_block = tl.program_id(2)
    h_out_block = tl.program_id(3)
    w_out_block = tl.program_id(4)
    
    # Calculate starting positions for this block
    c_out_start = c_out_block * BLOCK_SIZE_C_out
    d_out_start = d_out_block * BLOCK_SIZE_K
    h_out_start = h_out_block * BLOCK_SIZE_K
    w_out_start = w_out_block * BLOCK_SIZE_K
    
    # Create offset arrays for output dimensions
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_out)
    d_out_offsets = d_out_start + tl.arange(0, BLOCK_SIZE_K)
    h_out_offsets = h_out_start + tl.arange(0, BLOCK_SIZE_K)
    w_out_offsets = w_out_start + tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for output bounds
    c_out_mask = c_out_offsets < C_out
    d_out_mask = d_out_offsets < D_out
    h_out_mask = h_out_offsets < H_out
    w_out_mask = w_out_offsets < W_out
    
    # Initialize accumulators for output
    # We'll compute one accumulator per output channel
    acc = tl.zeros((BLOCK_SIZE_C_out, BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Iterate over input channels in blocks
    for c_in_block in range(0, C_in, BLOCK_SIZE_C_in):
        c_in_start = c_in_block * BLOCK_SIZE_C_in
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C_in)
        c_in_mask = c_in_offsets < C_in
        
        # Load input block: [BLOCK_SIZE_C_in, BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_SIZE_K]
        # Calculate corresponding input positions for each output position
        for kd in range(BLOCK_SIZE_K):
            d_out = d_out_start + kd
            if d_out < D_out:
                # Calculate input depth index
                d_in = d_out - padding_d + kd * dilation_d
                d_in = (d_in - 1) // stride_d + 1  # This is incorrect, need to recalculate
                
        # Actually, let's do it more systematically
        # For each output position (d_out, h_out, w_out) and each kernel position (kd, kh, kw)
        # We need to compute: input_pos = (d_out - padding_d - kd * dilation_d) // stride_d + padding_d
        
        # Instead, let's restructure: iterate over kernel positions first
        
    # Let's rewrite with a better approach: iterate over kernel positions
    # and accumulate contributions to the output
    
    # Reset accumulators
    acc = tl.zeros((BLOCK_SIZE_C_out, BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Iterate over kernel positions
    for kd in range(K_d):
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate input position corresponding to this kernel position
                # For output position (d_out, h_out, w_out), the contribution comes from:
                # d_in = (d_out - padding_d - kd * dilation_d) // stride_d + padding_d
                # But we need to handle the batch dimension properly
                
                # Get input indices for current kernel position
                d_in_base = d_out_start - padding_d + kd * dilation_d
                h_in_base = h_out_start - padding_h + kh * dilation_h
                w_in_base = w_out_start - padding_w + kw * dilation_w
                
                # Iterate over output positions within our block
                for bd in range(BLOCK_SIZE_K):
                    d_out = d_out_start + bd
                    d_in = (d_out - padding_d - kd * dilation_d)
                    if d_in >= 0 and d_in % stride_d == 0:
                        d_in = d_in // stride_d
                        if d_in < D_in:
                            for bh in range(BLOCK_SIZE_K):
                                h_out = h_out_start + bh
                                h_in = (h_out - padding_h - kh * dilation_h)
                                if h_in >= 0 and h_in % stride_h == 0:
                                    h_in = h_in // stride_h
                                    if h_in < H_in:
                                        for bw in range(BLOCK_SIZE_K):
                                            w_out = w_out_start + bw
                                            w_in = (w_out - padding_w - kw * dilation_w)
                                            if w_in >= 0 and w_in % stride_w == 0:
                                                w_in = w_in // stride_w
                                                if w_in < W_in:
                                                    # Load input value at (batch_idx, :, d_in, h_in, w_in)
                                                    x_offset = batch_idx * (C_in * D_in * H_in * W_in) + \
                                                               tl.arange(0, BLOCK_SIZE_C_in) * (D_in * H_in * W_in) + \
                                                               d_in * (H_in * W_in) + \
                                                               h_in * W_in + \
                                                               w_in
                                                    x_mask = c_in_offsets < C_in
                                                    x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
                                                    
                                                    # Load weight value at (:, c_out, kd, kh, kw)
                                                    w_offset = tl.arange(0, BLOCK_SIZE_C_in) * (C_out * K_d * K_h * K_w) + \
                                                               c_out_offsets * (K_d * K_h * K_w) + \
                                                               kd * (K_h * K_w) + \
                                                               kh * K_w + \
                                                               kw
                                                    w_mask_c_in = tl.arange(0, BLOCK_SIZE_C_in) < C_in
                                                    w_mask_c_out = c_out_offsets < C_out
                                                    w_vals = tl.load(w_ptr + w_offset, mask=tl.logical_and(w_mask_c_in[:, None], w_mask_c_out[None, :]), other=0.0)
                                                    
                                                    # Accumulate: x_vals[BLOCK_SIZE_C_in] * w_vals[BLOCK_SIZE_C_in, BLOCK_SIZE_C_out]
                                                    # Result shape: [BLOCK_SIZE_C_in, BLOCK_SIZE_C_out]
                                                    # But we need [BLOCK_SIZE_C_out] per output position
                                                    # Actually, for this output position, we need to accumulate over c_in
                                                    # acc[c_out_idx] += sum_cin(x[c_in] * w[c_in, c_out])
                                                    
                                                    # Reshape for broadcasting: x_vals[BLOCK_SIZE_C_in, 1], w_vals[BLOCK_SIZE_C_in, BLOCK_SIZE_C_out]
                                                    # Result: [BLOCK_SIZE_C_in, BLOCK_SIZE_C_out]
                                                    # Then sum over c_in dimension to get [BLOCK_SIZE_C_out]
                                                    contrib = tl.sum(x_vals[:, None] * w_vals, axis=0)
                                                    acc[:, bd, bh, bw] += contrib
    
    # Add bias if enabled
    if HAS_BIAS:
        bias_offsets = tl.arange(0, BLOCK_SIZE_C_out)
        bias_mask = bias_offsets < C_out
        bias_vals = tl.load(b_ptr + bias_offsets, mask=bias_mask, other=0.0)
        acc += bias_vals[None, :, None, None]
    
    # Store results
    out_offsets = batch_idx * (C_out * D_out * H_out * W_out) + \
                  c_out_offsets[:, None, None, None] * (D_out * H_out * W_out) + \
                  (d_out_start + tl.arange(0, BLOCK_SIZE_K)[None, :, None, None]) * (H_out * W_out) + \
                  (h_out_start + tl.arange(0, BLOCK_SIZE_K)[None, None, :, None]) * W_out + \
                  (w_out_start + tl.arange(0, BLOCK_SIZE_K)[None, None, None, :])
    
    # Reshape acc to match output format: [BLOCK_SIZE_C_out, BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_SIZE_K]
    acc_flat = acc.reshape(BLOCK_SIZE_C_out * BLOCK_SIZE_K * BLOCK_SIZE_K * BLOCK_SIZE_K)
    out_offsets_flat = out_offsets.reshape(BLOCK_SIZE_C_out * BLOCK_SIZE_K * BLOCK_SIZE_K * BLOCK_SIZE_K)
    
    # Create masks for the flattened output
    out_mask = (c_out_offsets[:, None, None, None] < C_out) & \
               ((d_out_start + tl.arange(0, BLOCK_SIZE_K)[None, :, None, None]) < D_out) & \
               ((h_out_start + tl.arange(0, BLOCK_SIZE_K)[None, None, :, None]) < H_out) & \
               ((w_out_start + tl.arange(0, BLOCK_SIZE_K)[None, None, None, :]) < W_out)
    out_mask_flat = out_mask.reshape(BLOCK_SIZE_C_out * BLOCK_SIZE_K * BLOCK_SIZE_K * BLOCK_SIZE_K)
    
    tl.store(out_ptr + out_offsets_flat, acc_flat, mask=out_mask_flat)


def triton_conv_transpose3d(x, weight, bias, stride, padding, dilation):
    """
    Performs 3D transposed convolution using Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    N, C_in, D_in, H_in, W_in = x.shape
    C_in2, C_out, K_d, K_h, K_w = weight.shape
    assert C_in == C_in2, "Input channels must match"
    
    # Calculate output dimensions
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_d - 1) + 1
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_h - 1) + 1
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + dilation[2] * (K_w - 1) + 1
    
    # Allocate output tensor
    out = torch.empty(N, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Grid dimensions
    # We'll use a 5D grid: (batch, c_out_block, d_block, h_block, w_block)
    BLOCK_SIZE_C_out = 32
    BLOCK_SIZE_K = 2  # Small block size for kernel dimension
    
    grid_d = (D_out + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid_h = (H_out + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid_w = (W_out + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    grid = (N, (C_out + BLOCK_SIZE_C_out - 1) // BLOCK_SIZE_C_out, grid_d, grid_h, grid_w)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K_d, K_h, K_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        HAS_BIAS=bias is not None,
        BLOCK_SIZE_C_out=BLOCK_SIZE_C_out,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 3D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the same convolution layer but replace forward pass with Triton
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, 
                                                   kernel_size=(kernel_size, kernel_size, kernel_size), 
                                                   stride=stride, padding=padding, 
                                                   dilation=dilation, bias=bias)
        # Store parameters for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution using Triton kernel.
        """
        # Use our Triton implementation instead of PyTorch's
        return triton_conv_transpose3d(
            x, 
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias if self.has_bias else None,
            (self.stride, self.stride, self.stride),
            (self.padding, self.padding, self.padding),
            (self.dilation, self.dilation, self.dilation)
        )