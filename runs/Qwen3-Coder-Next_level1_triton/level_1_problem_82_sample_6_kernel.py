import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,              # Input tensor pointer (B, C, H, W)
    w_ptr,              # Weight tensor pointer (C, K, K)
    b_ptr,              # Bias tensor pointer (C,) or None
    out_ptr,            # Output tensor pointer (B, C, H_out, W_out)
    batch_size,         # Batch size
    in_channels,        # Number of input channels
    height,             # Input height
    width,              # Input width
    kernel_size,        # Kernel size
    stride,             # Stride
    padding,            # Padding
    out_height,         # Output height
    out_width,          # Output width
    BLOCK_SIZE_C: tl.constexpr,  # Block size for channels
    BLOCK_SIZE_H: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_W: tl.constexpr,  # Block size for output width
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
):
    # Get program IDs
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Calculate channel block start
    c_start = pid_c * BLOCK_SIZE_C
    c_mask = c_start + tl.arange(0, BLOCK_SIZE_C) < in_channels
    
    # Calculate output spatial positions
    h_out_start = pid_h * BLOCK_SIZE_H
    w_out_start = pid_w * BLOCK_SIZE_W
    
    # Output offsets for this block
    out_offsets_h = h_out_start + tl.arange(0, BLOCK_SIZE_H)[:, None]
    out_offsets_w = w_out_start + tl.arange(0, BLOCK_SIZE_W)[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Convolution loop over kernel spatial dimensions
    for kh in range(KERNEL_SIZE):
        for kw in range(KERNEL_SIZE):
            # Calculate input spatial position
            h_in = h_out_start * STRIDE + kh - PADDING
            w_in = w_out_start * STRIDE + kw - PADDING
            
            # Check if input position is valid
            valid_h = (h_in >= 0) & (h_in < height)
            valid_w = (w_in >= 0) & (w_in < width)
            
            if valid_h and valid_w:
                # Calculate input offsets
                # Input shape: (batch_size, in_channels, height, width)
                h_in_offset = h_in
                w_in_offset = w_in
                
                # Load input values for this kernel position
                # We need to handle the batch dimension as well
                for b in range(batch_size):
                    # Input tensor layout: (B, C, H, W)
                    # Calculate base offset for this batch
                    batch_offset = b * in_channels * height * width
                    
                    # Calculate input channel offsets
                    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
                    
                    # Create masks for valid channels and positions
                    mask_c = c_offsets < in_channels
                    mask = mask_c[:, None, None]  # (BLOCK_SIZE_C, 1, 1)
                    
                    # Calculate input pointer offset for this batch
                    input_offset = batch_offset + h_in_offset * width + w_in_offset
                    
                    # Load input: (B, C, H, W) -> need to slice properly
                    # For simplicity, we'll process one batch at a time
                    
                    # Load input slice for current batch
                    # We need to handle the actual memory layout carefully
                    # Since we're doing depthwise, we can load channel-wise
                    
                    # For each channel in the block
                    for i_c in range(BLOCK_SIZE_C):
                        c_idx = c_start + i_c
                        if c_idx < in_channels:
                            # Input tensor is contiguous, so calculate offset
                            input_ptr_c = x_ptr + batch_offset + c_idx * height * width + h_in_offset * width + w_in_offset
                            x_val = tl.load(input_ptr_c)
                            
                            # Load weight for this channel and kernel position
                            weight_ptr_c = w_ptr + c_idx * KERNEL_SIZE * KERNEL_SIZE + kh * KERNEL_SIZE + kw
                            w_val = tl.load(weight_ptr_c)
                            
                            # Accumulate
                            acc[i_c, :, :] += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        bias_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
        bias_mask = bias_offsets < in_channels
        bias = tl.load(b_ptr + bias_offsets, mask=bias_mask, other=0.0)
        # Bias shape: (C,) -> need to broadcast to (C, H_out, W_out)
        bias = bias[:, None, None]  # (BLOCK_SIZE_C, 1, 1)
        acc += bias
    
    # Store output
    for b in range(batch_size):
        batch_offset = b * in_channels * out_height * out_width
        out_offsets_c = c_start + tl.arange(0, BLOCK_SIZE_C)
        out_mask_c = out_offsets_c < in_channels
        
        # Calculate output base offset for this batch
        out_base = batch_offset + out_offsets_h * out_width + out_offsets_w
        
        # Store results
        for i_c in range(BLOCK_SIZE_C):
            c_idx = c_start + i_c
            if c_idx < in_channels:
                out_ptr_c = out_ptr + batch_offset + c_idx * out_height * out_width
                out_offset_h = out_offsets_h
                out_offset_w = out_offsets_w
                out_value = acc[i_c, :, :]
                
                # Store with proper masking
                mask_h = out_offsets_h < out_height
                mask_w = out_offsets_w < out_width
                mask = mask_h & mask_w
                
                # We need to store element by element due to triton limitations
                for oh in range(BLOCK_SIZE_H):
                    for ow in range(BLOCK_SIZE_W):
                        if oh < out_height - h_out_start and ow < out_width - w_out_start:
                            tl.store(out_ptr_c + oh * out_width + ow, out_value[oh, ow])


# Alternative, more efficient implementation
@triton.jit
def depthwise_conv2d_v2_kernel(
    x_ptr,              # Input tensor (B, C, H, W) - contiguous
    w_ptr,              # Weights (C, K, K)
    b_ptr,              # Bias (C,) or None
    out_ptr,            # Output (B, C, H_out, W_out)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    P: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Channel block
    c_offset = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offset < C
    
    # Output spatial positions
    h_out = pid_h * BLOCK_H
    w_out = pid_w * BLOCK_W
    
    # Output tensor offsets for this block
    # We'll compute for one batch at a time in the outer loop
    
    # Load weights for this channel block
    # Weights shape: (C, K, K)
    w = tl.zeros((BLOCK_C, K, K), dtype=tl.float32)
    for kh in range(K):
        for kw in range(K):
            w_ptr_offset = c_offset * (K * K) + kh * K + kw
            w_val = tl.load(w_ptr + w_ptr_offset, mask=c_mask, other=0.0)
            w = tl.store(w + tl.where(tl.arange(0, BLOCK_C)[:, None, None] < C, 
                                      tl.full((BLOCK_C, 1, 1), kh * K + kw), 0),
                        w_val)  # This is incorrect, let me fix
    
    # Actually, let's load weights properly
    w = tl.zeros((BLOCK_C, K, K), dtype=tl.float32)
    for c_idx in range(BLOCK_C):
        c = pid_c * BLOCK_C + c_idx
        if c < C:
            for kh in range(K):
                for kw in range(K):
                    w_ptr_offset = c * (K * K) + kh * K + kw
                    w_val = tl.load(w_ptr + w_ptr_offset)
                    w = w.at[c_idx, kh, kw].set(w_val)
    
    # Load bias if available
    bias = tl.zeros((BLOCK_C,), dtype=tl.float32)
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_offset, mask=c_mask, other=0.0)
    
    # Process each batch
    for b in range(B):
        # Calculate input position for top-left of output
        h_in_start = h_out * S - P
        w_in_start = w_out * S - P
        
        # Accumulator for output
        out_acc = tl.zeros((BLOCK_C, BLOCK_H, BLOCK_W), dtype=tl.float32)
        
        # Convolution over kernel spatial dimensions
        for kh in range(K):
            for kw in range(K):
                h_in = h_in_start + kh
                w_in = w_in_start + kw
                
                # Check if input position is valid
                valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                
                if valid:
                    # Load input for this position
                    # Input tensor: (B, C, H, W)
                    input_base = b * C * H * W + h_in * W + w_in
                    
                    # Load input values for channel block
                    x_val = tl.zeros((BLOCK_C,), dtype=tl.float32)
                    for c_idx in range(BLOCK_C):
                        c = pid_c * BLOCK_C + c_idx
                        if c < C:
                            x_val = x_val.at[c_idx].set(tl.load(x_ptr + input_base + c * H * W))
                    
                    # Multiply with weights and accumulate
                    for c_idx in range(BLOCK_C):
                        if c_offset[c_idx] < C:
                            for oh in range(BLOCK_H):
                                for ow in range(BLOCK_W):
                                    h_out_local = h_out + oh
                                    w_out_local = w_out + ow
                                    # Check if output position is valid
                                    if h_out_local < H_out and w_out_local < W_out:
                                        h_in_offset = h_out_local * S - P + kh
                                        w_in_offset = w_out_local * S - P + kw
                                        if h_in_offset == h_in and w_in_offset == w_in:
                                            out_acc = out_acc.at[c_idx, oh, ow].add(
                                                x_val[c_idx] * w[c_idx, kh, kw]
                                            )
        
        # Add bias
        for c_idx in range(BLOCK_C):
            if c_offset[c_idx] < C:
                out_acc = out_acc.at[c_idx, :, :].add(bias[c_idx])
        
        # Store output
        output_base = b * C * H_out * W_out
        for c_idx in range(BLOCK_C):
            if c_offset[c_idx] < C:
                c = c_offset[c_idx]
                output_ptr_c = out_ptr + output_base + c * H_out * W_out
                for oh in range(BLOCK_H):
                    for ow in range(BLOCK_W):
                        if h_out + oh < H_out and w_out + ow < W_out:
                            tl.store(output_ptr_c + oh * W_out + ow, out_acc[c_idx, oh, ow])


# Even simpler and more efficient implementation
@triton.jit
def depthwise_conv2d_kernel_final(
    x_ptr,              # Input: (B, C, H, W)
    w_ptr,              # Weights: (C, K, K)
    b_ptr,              # Bias: (C,) or None
    out_ptr,            # Output: (B, C, H_out, W_out)
    B, C, H, W, K, S, P, H_out, W_out,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Get program IDs
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Calculate channel block
    c_start = pid_c * BLOCK_C
    c_range = tl.arange(0, BLOCK_C)
    c_mask = c_start + c_range < C
    
    # Calculate output spatial indices
    h_out = pid_h * BLOCK_H
    w_out = pid_w * BLOCK_W
    
    # Preload weights for this channel block
    weights = tl.zeros((BLOCK_C, K, K), dtype=tl.float32)
    for c_idx in range(BLOCK_C):
        c = c_start + c_idx
        if c < C:
            for kh in range(K):
                for kw in range(K):
                    w_idx = c * K * K + kh * K + kw
                    weights = weights.at[c_idx, kh, kw].set(tl.load(w_ptr + w_idx))
    
    # Preload bias if available
    bias_vals = tl.zeros((BLOCK_C,), dtype=tl.float32)
    if b_ptr is not None:
        bias_vals = tl.load(b_ptr + c_start + c_range, mask=c_mask, other=0.0)
    
    # Process each batch
    for b in range(B):
        # Initialize accumulator
        acc = tl.zeros((BLOCK_C, BLOCK_H, BLOCK_W), dtype=tl.float32)
        
        # Input base offset for this batch
        input_base = b * C * H * W
        
        # Convolution
        for kh in range(K):
            for kw in range(K):
                # Input position for top-left output
                h_in = h_out * S + kh - P
                w_in = w_out * S + kw - P
                
                # Only process if this kernel position affects valid output
                if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                    # Load input values for all channels in block
                    x_vals = tl.zeros((BLOCK_C,), dtype=tl.float32)
                    for c_idx in range(BLOCK_C):
                        c = c_start + c_idx
                        if c < C:
                            input_offset = input_base + c * H * W + h_in * W + w_in
                            x_vals = x_vals.at[c_idx].set(tl.load(x_ptr + input_offset))
                    
                    # Accumulate: x * w for each channel
                    for c_idx in range(BLOCK_C):
                        if c_start + c_idx < C:
                            for oh in range(BLOCK_H):
                                for ow in range(BLOCK_W):
                                    h_out_idx = h_out + oh
                                    w_out_idx = w_out + ow
                                    # Check if this output position is valid
                                    if h_out_idx < H_out and w_out_idx < W_out:
                                        # Calculate input position for this output
                                        h_in_pos = h_out_idx * S + kh - P
                                        w_in_pos = w_out_idx * S + kw - P
                                        if h_in_pos == h_in and w_in_pos == w_in:
                                            acc = acc.at[c_idx, oh, ow].add(
                                                x_vals[c_idx] * weights[c_idx, kh, kw]
                                            )
        
        # Add bias
        for c_idx in range(BLOCK_C):
            if c_start + c_idx < C:
                acc = acc.at[c_idx, :, :].add(bias_vals[c_idx])
        
        # Store output
        output_base = b * C * H_out * W_out
        for c_idx in range(BLOCK_C):
            c = c_start + c_idx
            if c < C:
                output_ptr_c = out_ptr + output_base + c * H_out * W_out
                for oh in range(BLOCK_H):
                    for ow in range(BLOCK_W):
                        h_out_idx = h_out + oh
                        w_out_idx = w_out + ow
                        if h_out_idx < H_out and w_out_idx < W_out:
                            tl.store(output_ptr_c + oh * W_out + ow, acc[c_idx, oh, ow])


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0):
    """Triton implementation of depthwise 2D convolution"""
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C, H, W = x.shape
    _, _, K, _ = weight.shape
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1
    
    # Allocate output
    out = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Configure block sizes
    BLOCK_C = 4  # Process 4 channels per block
    BLOCK_H = 8  # Process 8 output height positions per block  
    BLOCK_W = 8  # Process 8 output width positions per block
    
    # Grid dimensions
    grid = (
        (C + BLOCK_C - 1) // BLOCK_C,
        (H_out + BLOCK_H - 1) // BLOCK_H,
        (W_out + BLOCK_W - 1) // BLOCK_W
    )
    
    # Launch kernel
    depthwise_conv2d_kernel_final[grid](
        x, weight, bias, out,
        B, C, H, W, K, stride, padding, H_out, W_out,
        BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep the same structure as the original
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using our Triton kernel.
        """
        return triton_depthwise_conv2d(
            x, self.weight, self.bias, 
            stride=self.stride, padding=self.padding
        )