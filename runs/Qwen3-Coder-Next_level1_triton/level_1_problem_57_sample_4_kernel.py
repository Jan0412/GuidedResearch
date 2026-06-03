import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor: (N, C_in, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out // groups, kH, kW)
    b_ptr,  # Bias tensor: (C_out,) or None
    y_ptr,  # Output tensor: (N, C_out, H_out, W_out)
    N, C_in, C_out, groups,
    H, W,
    kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    out_pad_h, out_pad_w,
    H_out, W_out,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    
    # Skip if out of bounds
    if pid_batch >= N or pid_out_ch >= (C_out // BLOCK_SIZE_N):
        return
    
    # Compute output channel range for this block
    out_ch_start = pid_out_ch * BLOCK_SIZE_N
    out_ch_end = min(out_ch_start + BLOCK_SIZE_N, C_out)
    out_ch_range = out_ch_end - out_ch_start
    
    # Create output channel offsets
    out_ch_offsets = out_ch_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Iterate over input channels
    for in_ch in range(C_in):
        # Compute input position for each output position
        for kh in range(kH):
            for kw in range(kW):
                # Compute output position (h_out, w_out) that would receive contribution from (h_in, w_in)
                # For transposed conv: h_out = h_in * stride_h + kh - pad_h, similarly for w
                # So for a given output (h_out, w_out), the corresponding input is:
                # h_in = (h_out + pad_h - kh) // stride_h, w_in = (w_out + pad_w - kw) // stride_w
                
                # For efficiency, we'll compute the contribution to each output position
                # from the current (in_ch, kh, kw) weight
                
                # Compute weight value
                w_val = tl.load(w_ptr + in_ch * (C_out * kH * kW) + out_ch_offsets * (kH * kW) + kh * kW + kw)
                
                # Compute valid output positions for this (kh, kw)
                # h_out = h_in * stride_h + kh - pad_h
                # h_in must be in [0, H-1], so h_out in [kh - pad_h, (H-1)*stride_h + kh - pad_h]
                # But also need h_out in [0, H_out-1]
                for h_out in range(H_out):
                    # h_in = (h_out + pad_h - kh) / stride_h
                    h_in = (h_out + pad_h - kh) // stride_h
                    if h_in * stride_h + kh - pad_h != h_out:  # Check exact match
                        continue
                    if h_in < 0 or h_in >= H:
                        continue
                    
                    for w_out in range(W_out):
                        w_in = (w_out + pad_w - kw) // stride_w
                        if w_in * stride_w + kw - pad_w != w_out:
                            continue
                        if w_in < 0 or w_in >= W:
                            continue
                        
                        # Load input value
                        x_val = tl.load(x_ptr + pid_batch * (C_in * H * W) + in_ch * (H * W) + h_in * W + w_in)
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Apply bias if present
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + out_ch_offsets)
        acc += bias_val
    
    # Store result
    y_offsets = pid_batch * (C_out * H_out * W_out) + out_ch_offsets * (H_out * W_out)
    tl.store(y_ptr + y_offsets + h_out * W_out + w_out, acc.to(tl.float32))


# Better approach: use the standard approach for transposed conv which is equivalent to regular conv with different geometry
@triton.jit
def conv_transpose2d_fused_kernel(
    x_ptr,  # Input: (N, C_in, H, W)
    w_ptr,  # Weight: (C_in, C_out // groups, kH, kW)
    b_ptr,  # Bias: (C_out,) or None
    y_ptr,  # Output: (N, C_out, H_out, W_out)
    N, C_in, C_out, groups,
    H, W,
    kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    out_pad_h, out_pad_w,
    H_out, W_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Check bounds
    if pid_batch >= N or pid_out_ch >= (C_out // BLOCK_SIZE_COUT) or pid_h >= (H_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N or pid_w >= (W_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N:
        return
    
    # Compute actual positions
    batch_idx = pid_batch
    out_ch_start = pid_out_ch * BLOCK_SIZE_COUT
    out_ch_idx = out_ch_start + tl.arange(0, BLOCK_SIZE_COUT)
    h_idx = pid_h * BLOCK_SIZE_N
    w_idx = pid_w * BLOCK_SIZE_N
    
    # Check bounds for output channel and spatial positions
    mask_cout = out_ch_idx < C_out
    mask_h = h_idx < H_out
    mask_w = w_idx < W_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_COUT,), dtype=tl.float32)
    
    # Iterate over input channels, kernel height, kernel width
    for cin in range(C_in):
        for kh in range(kH):
            for kw in range(kW):
                # Compute corresponding input position
                h_in = (h_idx + pad_h - kh) // stride_h
                w_in = (w_idx + pad_w - kw) // stride_w
                
                # Check if valid input position
                valid_h = (h_in * stride_h + kh - pad_h == h_idx) & (h_in >= 0) & (h_in < H)
                valid_w = (w_in * stride_w + kw - pad_w == w_idx) & (w_in >= 0) & (w_in < W)
                valid = valid_h & valid_w
                
                # Load weight: shape (C_in, C_out, kH, kW) but stored as (C_in, C_out//groups, kH, kW) with groups
                weight_offset = cin * (C_out * kH * kW) + out_ch_idx * (kH * kW) + kh * kW + kw
                w_val = tl.load(w_ptr + weight_offset)
                
                # Load input: shape (N, C_in, H, W)
                input_offset = batch_idx * (C_in * H * W) + cin * (H * W) + h_in * W + w_in
                x_val = tl.load(x_ptr + input_offset) if valid else 0.0
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + out_ch_idx)
        acc += bias_val
    
    # Store result
    output_offset = batch_idx * (C_out * H_out * W_out) + out_ch_idx * (H_out * W_out) + h_idx * W_out + w_idx
    tl.store(y_ptr + output_offset, acc.to(tl.float32), mask=mask_cout & mask_h[:, None] & mask_w[None, :])


# Optimized implementation using matrix multiplication approach
@triton.jit
def conv_transpose2d_matmul_kernel(
    x_ptr,  # Input: (N, C_in, H, W)
    w_ptr,  # Weight: (C_in, C_out // groups, kH, kW)
    b_ptr,  # Bias: (C_out,) or None
    y_ptr,  # Output: (N, C_out, H_out, W_out)
    N, C_in, C_out, groups,
    H, W,
    kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    out_pad_h, out_pad_w,
    H_out, W_out,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch * H_out * W_out
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for C_in * kH * kW
):
    # Get program IDs
    pid = tl.program_id(0)
    pid_cout = tl.program_id(1)
    
    # Compute which (batch, h_out, w_out) this block handles
    spatial_size = H_out * W_out
    batch_spatial_idx = pid * BLOCK_SIZE_M
    batch_spatial_end = min(batch_spatial_idx + BLOCK_SIZE_M, N * spatial_size)
    
    # Compute output channel block
    cout_start = pid_cout * BLOCK_SIZE_N
    cout_end = min(cout_start + BLOCK_SIZE_N, C_out)
    cout_range = cout_end - cout_start
    
    # Create offsets
    cout_offsets = cout_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Iterate over K dimension (C_in * kH * kW)
    for k in range(C_in * kH * kW):
        # Extract cin, kh, kw from k
        cin = k // (kH * kW)
        remainder = k % (kH * kW)
        kh = remainder // kW
        kw = remainder % kW
        
        # For each spatial position in output
        for i in range(batch_spatial_idx, min(batch_spatial_idx + BLOCK_SIZE_M, N * spatial_size)):
            batch_idx = i // spatial_size
            spatial_idx = i % spatial_size
            h_out = spatial_idx // W_out
            w_out = spatial_idx % W_out
            
            # Compute input position for transposed convolution
            h_in = (h_out + pad_h - kh) // stride_h
            w_in = (w_out + pad_w - kw) // stride_w
            
            # Check if valid
            valid = ((h_in * stride_h + kh - pad_h == h_out) and 
                    (w_in * stride_w + kw - pad_w == w_out) and 
                    h_in >= 0 and h_in < H and 
                    w_in >= 0 and w_in < W)
            
            # Load input value
            x_offset = batch_idx * (C_in * H * W) + cin * (H * W) + h_in * W + w_in
            x_val = tl.load(x_ptr + x_offset) if valid else 0.0
            
            # Load weight value
            w_offset = cin * (C_out * kH * kW) + cout_offsets * (kH * kW) + kh * kW + kw
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            acc += x_val * w_val
    
    # Apply bias if present
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + cout_offsets)
        acc += bias_val
    
    # Store results
    for i in range(batch_spatial_idx, min(batch_spatial_idx + BLOCK_SIZE_M, N * spatial_size)):
        batch_idx = i // spatial_size
        spatial_idx = i % spatial_size
        h_out = spatial_idx // W_out
        w_out = spatial_idx % W_out
        
        y_offset = batch_idx * (C_out * H_out * W_out) + cout_offsets * (H_out * W_out) + h_out * W_out + w_out
        tl.store(y_ptr + y_offset, acc.to(tl.float32))


def triton_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups):
    """
    Performs transposed 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (N, C_in, H, W)
        weight: Weight tensor of shape (C_in, C_out // groups, kH, kW)
        bias: Bias tensor of shape (C_out,) or None
        stride: Tuple (stride_h, stride_w)
        padding: Tuple (pad_h, pad_w)
        output_padding: Tuple (out_pad_h, out_pad_w)
        groups: Number of groups
    
    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    N, C_in, H, W = x.shape
    C_out = weight.shape[1] * groups
    kH, kW = weight.shape[2], weight.shape[3]
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_pad_h, out_pad_w = output_padding
    
    # Calculate output dimensions
    H_out = (H - 1) * stride_h - 2 * pad_h + out_pad_h + (kH - 1) + 1
    W_out = (W - 1) * stride_w - 2 * pad_w + out_pad_w + (kW - 1) + 1
    
    # Prepare output tensor
    y = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Block sizes for parallelization
    BLOCK_SIZE_M = 64  # Batch * spatial size per block
    BLOCK_SIZE_N = 32  # Output channels per block
    BLOCK_SIZE_K = 16  # Accumulation dimension
    
    # Grid dimensions
    grid_m = (N * H_out * W_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    grid = (grid_m, grid_n)
    
    # Launch kernel
    conv_transpose2d_matmul_kernel[grid](
        x, weight, bias, y,
        N, C_in, C_out, groups,
        H, W,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        H_out, W_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Create the weight and bias parameters
        # Weight shape: (in_channels, out_channels // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights (same as PyTorch default initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using our optimized Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )


import math