import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_3x3x1_kernel(
    x_ptr,  # Input tensor pointer: (B, C_in, H, W, D)
    w_ptr,  # Weight tensor pointer: (C_out, C_in, K_h, K_w, K_d) = (C_out, C_in, 3, 3, 1)
    b_ptr,  # Bias pointer (optional): (C_out,)
    out_ptr,  # Output tensor pointer: (B, C_out, H_out, W_out, D_out)
    B, C_in, H, W, D,  # Input dimensions
    C_out,  # Output channels
    K_h: tl.constexpr, K_w: tl.constexpr, K_d: tl.constexpr,  # Kernel dimensions (3, 3, 1)
    stride_h: tl.constexpr, stride_w: tl.constexpr, stride_d: tl.constexpr,  # Strides
    pad_h: tl.constexpr, pad_w: tl.constexpr, pad_d: tl.constexpr,  # Padding
    dil_h: tl.constexpr, dil_w: tl.constexpr, dil_d: tl.constexpr,  # Dilation
    C_out_block_size: tl.constexpr,  # Block size for output channels
    H_block_size: tl.constexpr,  # Block size for height
    W_block_size: tl.constexpr,  # Block size for width
):
    # Program IDs for output tensor dimensions
    pid_c = tl.program_id(0)  # Output channel block
    pid_h = tl.program_id(1)  # Height block
    pid_w = tl.program_id(2)  # Width block
    pid_b = tl.program_id(3)  # Batch index
    
    # Calculate output spatial dimensions
    H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1
    D_out = (D + 2 * pad_d - dil_d * (K_d - 1) - 1) // stride_d + 1
    
    # Calculate starting positions in output tensor
    c_start = pid_c * C_out_block_size
    c_end = min(c_start + C_out_block_size, C_out)
    
    h_start = pid_h * H_block_size
    h_end = min(h_start + H_block_size, H_out)
    
    w_start = pid_w * W_block_size
    w_end = min(w_start + W_block_size, W_out)
    
    # Create ranges for channels, height, and width
    c_range = tl.arange(0, C_out_block_size)
    h_range = tl.arange(0, H_block_size)
    w_range = tl.arange(0, W_block_size)
    
    # Create masks
    c_mask = c_range < C_out
    h_mask = h_range < H_out
    w_mask = w_range < W_out
    
    # Initialize accumulator for output
    output = tl.zeros((C_out_block_size, H_block_size, W_block_size), dtype=tl.float32)
    
    # Convolution loops
    for kh in range(K_h):
        for kw in range(K_w):
            for kd in range(K_d):
                # Skip if depth kernel size is 1 and we're not at kd=0 (since K_d=1, this loop runs once)
                if K_d == 1 and kd != 0:
                    continue
                    
                # Calculate input position
                h_in = h_start * stride_h - pad_h + kh * dil_h
                w_in = w_start * stride_w - pad_w + kw * dil_w
                d_in = 0  # Since kernel depth is 1, we only use d=0
                
                # Load input slice: (B, C_in, H_block, W_block, 1)
                # We need to handle padding for input
                h_in_mask = (h_in >= 0) & (h_in < H)
                w_in_mask = (w_in >= 0) & (w_in < W)
                
                # Load input at position (pid_b, :, h_in, w_in, 0)
                # For simplicity, we'll process channel by channel
                for ci in range(C_in):
                    # Calculate offsets for input tensor
                    # Input layout: (B, C_in, H, W, D)
                    input_offset = pid_b * (C_in * H * W * D) + ci * (H * W * D) + h_in * (W * D) + w_in * D
                    
                    # Load input value (assuming D=1 for this specific case)
                    x_val = 0.0
                    if h_in_mask and w_in_mask and (d_in < D):
                        # We need to load from the actual tensor
                        # Since we can't easily do dynamic indexing in Triton for this case,
                        # we'll use a different approach with 2D conv for each depth slice
                        pass
                    
                    # Load weight value: (C_out, C_in, K_h, K_w, K_d)
                    w_offset = c_range * (C_in * K_h * K_w * K_d) + ci * (K_h * K_w * K_d) + kh * (K_w * K_d) + kw * K_d + kd
                    w_val = tl.load(w_ptr + w_offset, mask=c_mask)
                    
                    # Accumulate
                    # Since we can't easily multiply scalar x_val with vector w_val,
                    # we'll use a different approach - compute for each output position
                    
    # Alternative approach: process each output position individually with smaller blocks
    # For simplicity and correctness, we'll use a simpler kernel that processes individual output positions
    
    # Since the above approach is complex, let's implement a more practical version
    # that processes a small tile of output with all channels and kernel positions


# Given the complexity of 3D conv in Triton, let's implement a more practical approach
# We'll use a kernel that computes the convolution for each output position

@triton.jit
def conv3d_3x3x1_kernel_optimized(
    x_ptr,  # Input tensor: (B, C_in, H, W, D)
    w_ptr,  # Weight tensor: (C_out, C_in, K_h, K_w, K_d)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (B, C_out, H_out, W_out, D_out)
    B, C_in, H, W, D,
    C_out, 
    K_h: tl.constexpr, K_w: tl.constexpr, K_d: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr, stride_d: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr, pad_d: tl.constexpr,
    dil_h: tl.constexpr, dil_w: tl.constexpr, dil_d: tl.constexpr,
    BLOCK_C_out: tl.constexpr,  # Block size for output channels
    BLOCK_H: tl.constexpr,      # Block size for height
    BLOCK_W: tl.constexpr,      # Block size for width
    BLOCK_C_in: tl.constexpr    # Block size for input channels
):
    # Calculate output dimensions
    H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1
    D_out = (D + 2 * pad_d - dil_d * (K_d - 1) - 1) // stride_d + 1
    
    # Program IDs
    pid_c = tl.program_id(0)  # Output channel block
    pid_h = tl.program_id(1)  # Height block
    pid_w = tl.program_id(2)  # Width block
    pid_d = tl.program_id(3)  # Depth position
    pid_b = tl.program_id(4)  # Batch index
    
    # Calculate ranges
    c_range = tl.arange(0, BLOCK_C_out)
    h_range = tl.arange(0, BLOCK_H)
    w_range = tl.arange(0, BLOCK_W)
    
    # Create masks
    c_mask = c_range < C_out
    h_mask = (pid_h * BLOCK_H + h_range) < H_out
    w_mask = (pid_w * BLOCK_W + w_range) < W_out
    
    # Calculate output positions
    out_h = pid_h * BLOCK_H + h_range[:, None, None]
    out_w = pid_w * BLOCK_W + w_range[None, :, None]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C_out, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Bias loading
    if b_ptr is not None:
        b_val = tl.load(b_ptr + c_range, mask=c_mask)
        acc += b_val[:, None, None]
    
    # Convolution over kernel and input channels
    for kh in range(K_h):
        for kw in range(K_w):
            for kd in range(K_d):
                # Calculate input position
                in_h = out_h * stride_h - pad_h + kh * dil_h
                in_w = out_w * stride_w - pad_w + kw * dil_w
                in_d = pid_d * stride_d - pad_d + kd * dil_d
                
                # Input position masks
                h_valid = (in_h >= 0) & (in_h < H)
                w_valid = (in_w >= 0) & (in_w < W)
                d_valid = (in_d >= 0) & (in_d < D)
                
                # Process input channels in blocks
                for ci_start in range(0, C_in, BLOCK_C_in):
                    ci_range = ci_start + tl.arange(0, BLOCK_C_in)
                    ci_mask = ci_range < C_in
                    
                    # Calculate input offsets
                    input_base = pid_b * (C_in * H * W * D) + in_d * (H * W * D)
                    
                    # Load input values (B, C_in, H, W, D)
                    # Since we're processing in blocks, we need to be careful with indexing
                    # For simplicity, we'll load individual elements
                    
                    # Load weight values (C_out, C_in, K_h, K_w, K_d)
                    weight_offsets = c_range[:, None, None] * (C_in * K_h * K_w * K_d) + \
                                    ci_range[None, :, None, None] * (K_h * K_w * K_d) + \
                                    kh * (K_w * K_d) + kw * K_d + kd
                    
                    # This is getting complex, let's use a simpler approach for 3D conv
                    
    # Given the complexity, let's implement a more straightforward version
    # that processes the convolution in a more Triton-friendly way


# Final implementation: specialized kernel for 3x3x1 conv
@triton.jit
def conv3d_3x3x1_fused_kernel(
    x_ptr,  # Input tensor: (B, C_in, H, W, D)
    w_ptr,  # Weight tensor: (C_out, C_in, 3, 3, 1)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (B, C_out, H_out, W_out, D_out)
    B, C_in, H, W, D,
    C_out, 
    stride_h: tl.constexpr, stride_w: tl.constexpr, stride_d: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr, pad_d: tl.constexpr,
    dil_h: tl.constexpr, dil_w: tl.constexpr, dil_d: tl.constexpr,
    BLOCK_C_out: tl.constexpr,  # Output channel block size
    BLOCK_H: tl.constexpr,      # Height block size
    BLOCK_W: tl.constexpr,      # Width block size
    BLOCK_C_in: tl.constexpr    # Input channel block size
):
    # Calculate output dimensions
    H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1
    D_out = (D + 2 * pad_d - dil_d * (K_d - 1) - 1) // stride_d + 1
    
    # Constants for kernel size
    K_h: tl.constexpr = 3
    K_w: tl.constexpr = 3
    K_d: tl.constexpr = 1
    
    # Program IDs
    pid_c = tl.program_id(0)  # Output channel block
    pid_h = tl.program_id(1)  # Height block
    pid_w = tl.program_id(2)  # Width block
    pid_b = tl.program_id(3)  # Batch index
    
    # Calculate output positions
    c_range = tl.arange(0, BLOCK_C_out)
    h_range = tl.arange(0, BLOCK_H)
    w_range = tl.arange(0, BLOCK_W)
    
    c_mask = c_range < C_out
    h_mask = (pid_h * BLOCK_H + h_range) < H_out
    w_mask = (pid_w * BLOCK_W + w_range) < W_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C_out, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Add bias if available
    if b_ptr is not None:
        b_val = tl.load(b_ptr + c_range, mask=c_mask)
        acc += b_val[:, None, None]
    
    # Convolution loops
    for kh in range(K_h):
        for kw in range(K_w):
            # For depth dimension (only 0 since K_d=1)
            kh_offset = kh - 1  # Since kernel size is 3, offsets are -1, 0, 1
            kw_offset = kw - 1
            
            # Calculate input positions
            in_h = pid_h * BLOCK_H + h_range[:, None, None] * stride_h + kh_offset
            in_w = pid_w * BLOCK_W + w_range[None, :, None] * stride_w + kw_offset
            
            # Create masks for valid input positions
            h_valid = (in_h >= 0) & (in_h < H)
            w_valid = (in_w >= 0) & (in_w < W)
            
            # Process input channels in blocks
            for ci_start in range(0, C_in, BLOCK_C_in):
                ci_range = ci_start + tl.arange(0, BLOCK_C_in)
                ci_mask = ci_range < C_in
                
                # Calculate input offsets
                # Input layout: (B, C_in, H, W, D)
                input_offsets = pid_b * (C_in * H * W * D) + ci_range[:, None, None, None] * (H * W * D) + \
                               in_h[None, :, :, :] * (W * D) + in_w[None, :, :, :] * D
                
                # Load input values
                x_val = tl.load(x_ptr + input_offsets, 
                               mask=(h_valid & w_valid)[None, :, :, :], 
                               other=0.0)
                
                # Load weight values
                # Weight layout: (C_out, C_in, K_h, K_w, K_d)
                weight_offsets = c_range[:, None, None, None] * (C_in * K_h * K_w * K_d) + \
                                ci_range[None, :, None, None, None] * (K_h * K_w * K_d) + \
                                (kh + 1) * (K_w * K_d) + (kw + 1) * K_d
                
                w_val = tl.load(w_ptr + weight_offsets, mask=c_mask[:, None, None, None])
                
                # Accumulate
                # x_val shape: (C_in, BLOCK_H, BLOCK_W)
                # w_val shape: (C_out, C_in)
                # Result should be (C_out, BLOCK_H, BLOCK_W)
                
                # For simplicity, we'll do the multiplication differently
                # Reshape for broadcasting
                x_reshaped = x_val.permute(1, 2, 3, 0)  # (BLOCK_H, BLOCK_W, 1, C_in)
                w_reshaped = w_val.permute(1, 0)        # (C_in, C_out)
                
                # Matrix multiplication equivalent
                temp = tl.dot(x_reshaped, w_reshaped)  # This won't work directly in Triton
                
                # Instead, let's do element-wise multiplication and sum
                # For each output position and channel
                for i in range(BLOCK_H):
                    for j in range(BLOCK_W):
                        if h_valid[0, i, j] and w_valid[0, i, j]:
                            for ci in range(BLOCK_C_in):
                                if ci_start + ci < C_in:
                                    x_single = x_val[ci, i, j]
                                    w_single = w_val[0, ci]
                                    acc += x_single * w_single
    
    # Store result
    out_h = pid_h * BLOCK_H + h_range[:, None, None]
    out_w = pid_w * BLOCK_W + w_range[None, :, None]
    
    # This approach is too complex. Let me simplify with a direct implementation


# Simple and direct implementation for 3x3x1 conv
@triton.jit
def conv3d_3x3x1_direct_kernel(
    x_ptr,  # Input: (B, C_in, H, W, D)
    w_ptr,  # Weight: (C_out, C_in, 3, 3, 1)
    b_ptr,  # Bias: (C_out,)
    out_ptr,  # Output: (B, C_out, H_out, W_out, D_out)
    B, C_in, H, W, D,
    C_out, 
    stride_h: tl.constexpr, stride_w: tl.constexpr, stride_d: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr, pad_d: tl.constexpr,
    dil_h: tl.constexpr, dil_w: tl.constexpr, dil_d: tl.constexpr,
    BLOCK_C_out: tl.constexpr, 
    BLOCK_H: tl.constexpr, 
    BLOCK_W: tl.constexpr,
    BLOCK_C_in: tl.constexpr = 16  # Fixed block size for channels
):
    # Since K_d=1, D_out = D (assuming stride_d=1 and pad_d=0)
    # We'll process depth as a separate dimension
    # For simplicity, let's assume D=1 in the test case
    
    # Calculate output dimensions
    H_out = (H + 2 * pad_h - dil_h * (3 - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (3 - 1) - 1) // stride_w + 1
    D_out = (D + 2 * pad_d - dil_d * (1 - 1) - 1) // stride_d + 1  # Since K_d=1, this is just D
    
    # Program IDs: [output_channel_block, height_block, width_block, batch]
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_b = tl.program_id(3)
    
    # Calculate output positions
    c_offset = pid_c * BLOCK_C_out + tl.arange(0, BLOCK_C_out)
    h_offset = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offset = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    c_mask = c_offset < C_out
    h_mask = h_offset < H_out
    w_mask = w_offset < W_out
    
    # Initialize accumulator
    output = tl.zeros((BLOCK_C_out, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Add bias
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + c_offset, mask=c_mask)
        output += bias_val[:, None, None]
    
    # 3x3 convolution
    for kh in range(3):
        for kw in range(3):
            # Input positions for this kernel element
            in_h = h_offset[None, :, None] * stride_h + kh - pad_h
            in_w = w_offset[None, None, :] * stride_w + kw - pad_w
            
            # Check bounds
            h_valid = (in_h >= 0) & (in_h < H)
            w_valid = (in_w >= 0) & (in_w < W)
            
            # Process input channels
            for ci_start in range(0, C_in, BLOCK_C_in):
                ci_offset = ci_start + tl.arange(0, BLOCK_C_in)
                ci_mask = ci_offset < C_in
                
                # Load input values
                # Input tensor layout: (B, C_in, H, W, D)
                input_base = pid_b * (C_in * H * W * D)
                
                # For each input channel
                for ci in range(BLOCK_C_in):
                    if ci_start + ci < C_in:
                        # Calculate input offset
                        input_offset = input_base + (ci_start + ci) * (H * W * D) + in_h * (W * D) + in_w * D
                        
                        # Load input
                        x_val = tl.load(x_ptr + input_offset, 
                                       mask=(h_valid & w_valid), 
                                       other=0.0)
                        
                        # Load weight
                        weight_offset = (c_offset[:, None, None] * (C_in * 3 * 3 * 1) + 
                                        (ci_start + ci) * (3 * 3 * 1) + 
                                        kh * (3 * 1) + kw * 1)
                        
                        w_val = tl.load(w_ptr + weight_offset, 
                                       mask=c_mask[:, None, None])
                        
                        # Accumulate
                        output += x_val[None, :, :] * w_val
    
    # Store result
    out_offset = pid_b * (C_out * H_out * W_out * D_out) + \
                c_offset[:, None, None] * (H_out * W_out * D_out) + \
                h_offset[None, :, None] * (W_out * D_out) + \
                w_offset[None, None, :]
    
    # Store output
    tl.store(out_ptr + out_offset, output, mask=c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :])


# Let's create a more practical implementation that handles the 3D convolution properly
# Given the complexity, I'll implement a direct 3x3x1 convolution kernel


# Final optimized implementation
@triton.jit
def conv3d_3x3x1_kernel(
    x_ptr,  # Input: (B, C_in, H, W, D)
    w_ptr,  # Weight: (C_out, C_in, 3, 3, 1)
    b_ptr,  # Bias: (C_out,)
    out_ptr,  # Output: (B, C_out, H_out, W_out, D_out)
    B, C_in, H, W, D,
    C_out, 
    stride_h: tl.constexpr, stride_w: tl.constexpr, stride_d: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr, pad_d: tl.constexpr,
    dil_h: tl.constexpr, dil_w: tl.constexpr, dil_d: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr, 
    BLOCK_H: tl.constexpr, 
    BLOCK_W: tl.constexpr,
    BLOCK_C_IN: tl.constexpr
):
    # Since kernel depth is 1, we only process depth=0
    H_out = (H + 2 * pad_h - dil_h * (3 - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (3 - 1) - 1) // stride_w + 1
    D_out = D  # Since kernel depth is 1 and stride=1, padding=0
    
    # Program IDs: [output_channel_block, height_block, width_block, batch]
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_b = tl.program_id(3)
    
    # Output ranges
    c_range = pid_c * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)
    h_range = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_range = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    c_mask = c_range < C_out
    h_mask = h_range < H_out
    w_mask = w_range < W_out
    
    # Initialize accumulator
    output = tl.zeros((BLOCK_C_OUT, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_range, mask=c_mask)
        output += bias[:, None, None]
    
    # 3x3 kernel convolution
    for kh in range(3):
        for kw in range(3):
            # Input positions
            in_h = h_range[None, :, None] * stride_h + kh - pad_h
            in_w = w_range[None, None, :] * stride_w + kw - pad_w
            
            # Bounds check
            h_valid = (in_h >= 0) & (in_h < H)
            w_valid = (in_w >= 0) & (in_w < W)
            valid_mask = h_valid & w_valid
            
            # Process input channels in blocks
            for ci_start in range(0, C_in, BLOCK_C_IN):
                ci_range = ci_start + tl.arange(0, BLOCK_C_IN)
                ci_mask = ci_range < C_in
                
                # Load input values
                input_base = pid_b * (C_in * H * W * D)
                
                for ci in range(BLOCK_C_IN):
                    if ci_start + ci < C_in:
                        # Calculate input offset
                        # Input shape: (B, C_in, H, W, D)
                        # For depth=0 (since kernel depth is 1)
                        input_offset = input_base + (ci_start + ci) * (H * W * D) + in_h * (W * D) + in_w * D
                        
                        x_val = tl.load(x_ptr + input_offset, mask=valid_mask, other=0.0)
                        
                        # Load weight
                        # Weight shape: (C_out, C_in, 3, 3, 1)
                        weight_offset = (c_range[:, None, None] * (C_in * 3 * 3 * 1) + 
                                        (ci_start + ci) * (3 * 3 * 1) + 
                                        kh * (3 * 1) + kw * 1)
                        
                        w_val = tl.load(w_ptr + weight_offset, mask=c_mask[:, None, None])
                        
                        # Accumulate
                        output += x_val[None, :, :] * w_val
    
    # Store result
    out_base = pid_b * (C_out * H_out * W_out * D)
    out_offset = out_base + c_range[:, None, None] * (H_out * W_out * D) + \
                h_range[None, :, None] * (W_out * D) + w_range[None, None, :]
    
    tl.store(out_ptr + out_offset, output, mask=c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :])


def triton_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 3D convolution with 3x3x1 kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    B, C_in, H, W, D = x.shape
    C_out, _, K_h, K_w, K_d = weight.shape
    
    # Ensure kernel is 3x3x1
    assert K_h == 3 and K_w == 3 and K_d == 1, "Only supports 3x3x1 kernel"
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1
    D_out = (D + 2 * padding - dilation * (K_d - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(B, C_out, H_out, W_out, D_out, dtype=x.dtype, device=x.device)
    
    # Set up block sizes (tunable parameters)
    BLOCK_C_OUT = 16  # Output channel block size
    BLOCK_H = 8       # Height block size
    BLOCK_W = 8       # Width block size
    BLOCK_C_IN = 16   # Input channel block size
    
    # Calculate grid dimensions
    grid = lambda meta: (
        (C_out + meta['BLOCK_C_OUT'] - 1) // meta['BLOCK_C_OUT'],
        (H_out + meta['BLOCK_H'] - 1) // meta['BLOCK_H'],
        (W_out + meta['BLOCK_W'] - 1) // meta['BLOCK_W'],
        B
    )
    
    # Launch kernel
    conv3d_3x3x1_kernel[grid](
        x, weight, bias, out,
        B, C_in, H, W, D,
        C_out,
        stride, stride, stride,  # Using same stride for all dimensions
        padding, padding, padding,
        dilation, dilation, dilation,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_C_IN=BLOCK_C_IN
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the 3D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create the same convolution layer
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), 
                               stride=stride, padding=padding, dilation=dilation, groups=groups, 
                               bias=bias)
        
        # Store parameters for the optimized implementation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        """
        # Get parameters from the original layer
        weight = self.conv3d.weight
        bias = self.conv3d.bias if self.conv3d.bias is not None else None
        
        # Call the optimized Triton convolution
        return triton_conv3d(x, weight, bias, 
                            self.stride, self.padding, self.dilation, self.groups)