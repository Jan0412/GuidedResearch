import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out // groups, kH, kW)
    b_ptr,  # Bias tensor: (C_out,)
    y_ptr,  # Output tensor: (B, C_out, H_out, W_out)
    B, C_in, C_out, kH, kW, H, W, H_out, W_out, groups, g_cin, g_cout,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch * H_out * W_out
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels per group
):
    # Compute program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Total number of output elements per group
    total_elements = B * H_out * W_out
    
    # Compute which output element we're processing
    n_block_start = pid_n * BLOCK_SIZE_N
    offsets_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < total_elements
    
    # Convert linear index to (b, h_out, w_out)
    temp = offsets_n
    w_out_idx = temp % W_out
    temp = temp // W_out
    h_out_idx = temp % H_out
    b_idx = temp // H_out
    
    # Compute input spatial coordinates (considering stride and padding)
    h_in = h_out_idx * stride_h - padding_h + dilation_h * tl.arange(0, 1)
    w_in = w_out_idx * stride_w - padding_w + dilation_w * tl.arange(0, 1)
    
    # Compute output channel range
    m_block_start = pid_m * BLOCK_SIZE_M
    offsets_m = m_block_start + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < C_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels within the group
    for k in range(0, g_cin, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < g_cin
        
        # Compute input channel index in the full tensor
        # For each group, we have g_cin input channels and g_cout output channels
        group_idx = offsets_m // g_cout
        local_out_c = offsets_m % g_cout
        local_in_c = k_offsets
        
        # Compute full input channel indices
        in_c_offsets = group_idx * g_cin + local_in_c[:, None]
        in_c_offsets = tl.reshape(in_c_offsets, (BLOCK_SIZE_M, 1))
        
        # Compute kernel indices: weight[local_in_c, local_out_c, kh, kw]
        kh_offsets = tl.arange(0, kH)[:, None]
        kw_offsets = tl.arange(0, kW)[None, :]
        
        # Compute input position for each kernel position
        h_in_pos = h_out_idx * stride_h - padding_h + kh_offsets * dilation_h
        w_in_pos = w_out_idx * stride_w - padding_w + kw_offsets * dilation_w
        
        # Check if input position is valid
        valid_h = (h_in_pos >= 0) & (h_in_pos < H)
        valid_w = (w_in_pos >= 0) & (w_in_pos < W)
        valid = valid_h & valid_w
        
        # Load input values: shape (B, C_in, H, W)
        # We need to get x[b_idx, in_c_offsets, h_in_pos, w_in_pos]
        # This requires flattening and careful indexing
        
        # Simplified approach: process one element at a time for the spatial dimensions
        for batch_idx in range(B):
            # Compute input pointer offset for this batch
            x_batch_offset = batch_idx * C_in * H * W
            
            for kh in range(kH):
                for kw in range(kW):
                    h_in_pos_k = h_out_idx * stride_h - padding_h + kh * dilation_h
                    w_in_pos_k = w_out_idx * stride_w - padding_w + kw * dilation_w
                    
                    if h_in_pos_k >= 0 and h_in_pos_k < H and w_in_pos_k >= 0 and w_in_pos_k < W:
                        # Load input: x[b_idx, :, h_in_pos_k, w_in_pos_k]
                        x_offsets = x_batch_offset + in_c_offsets[:, 0] * H * W + h_in_pos_k * W + w_in_pos_k
                        x_val = tl.load(x_ptr + x_offsets, mask=mask_k, other=0.0)
                        
                        # Load weight: w[local_in_c, local_out_c, kh, kw]
                        # Weight layout: (C_in, C_out // groups, kH, kW)
                        w_offsets = local_in_c[:, None] * (C_out * kH * kW) + local_out_c[None, :] * (kH * kW) + kh * kW + kw
                        w_val = tl.load(w_ptr + w_offsets, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
                        
                        # Accumulate: acc += x_val * w_val
                        acc += x_val[:, None] * w_val
    
    # Add bias if present
    if HAS_BIAS:
        b_val = tl.load(b_ptr + offsets_m, mask=mask_m, other=0.0)
        acc += b_val[:, None]
    
    # Store output
    # Reshape acc to match output layout
    y_offsets = b_idx[:, None] * C_out * H_out * W_out + offsets_m[None, :] * H_out * W_out + h_out_idx * W_out + w_out_idx
    y_mask = mask_n & (mask_m[None, :])
    
    # Convert acc to float16 or float32 as needed
    tl.store(y_ptr + y_offsets, acc, mask=y_mask)


def triton_conv_transpose2d(x, weight, bias, stride, padding, dilation, groups):
    """
    Performs 2D transposed convolution using Triton kernel.
    """
    B, C_in, H, W = x.shape
    C_out = weight.shape[1] * groups  # weight shape: (C_in, C_out // groups, kH, kW)
    kH, kW = weight.shape[2], weight.shape[3]
    
    # Compute output dimensions
    H_out = (H - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kH - 1) + 1
    W_out = (W - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kW - 1) + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    y = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Set up kernel parameters
    g_cin = C_in // groups
    g_cout = C_out // groups
    
    # Kernel block sizes - tuned for FP32
    BLOCK_SIZE_M = 16  # output channels per block
    BLOCK_SIZE_N = 64  # spatial elements per block
    BLOCK_SIZE_K = 8   # input channels per block
    
    # Grid dimensions
    grid_m = (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (B * H_out * W_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out, kH, kW, H, W, H_out, W_out, groups, g_cin, g_cout,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        HAS_BIAS=bias is not None,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights: (C_in, C_out // groups, kH, kW)
        k = 1.0 / (in_channels * kernel_size[0] * kernel_size[1])
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, *kernel_size).uniform_(-math.sqrt(k), math.sqrt(k)))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).uniform_(-math.sqrt(k), math.sqrt(k)))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(x, self.weight, self.bias, 
                                       self.stride, self.padding, 
                                       self.dilation, self.groups)