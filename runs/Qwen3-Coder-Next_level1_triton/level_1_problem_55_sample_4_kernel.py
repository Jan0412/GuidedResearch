import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # [B, C_in, H, W]
    w_ptr,  # [C_out, C_in, K_h, K_w]
    b_ptr,  # [C_out] or None
    y_ptr,  # [B, C_out, H_out, W_out]
    # Dimensions
    B, C_in, H, W,
    C_out, K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    H_out, W_out,
    # Block sizes (tunable)
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    USE_MATRIX_MULTIPLY: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate base indices for output
    c_out_start = pid_c_out * BLOCK_C_out
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    
    # Create ranges for output dimensions
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_C_out)
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    
    # Create mask for valid indices
    c_out_mask = c_out_offsets < C_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C_out, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Convolution: iterate over input channels and kernel dimensions
    for c_in in range(0, C_in, BLOCK_C_in):
        c_in_start = c_in
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_C_in)
        c_in_mask = c_in_offsets < C_in
        
        for kh in range(0, K_h, BLOCK_K):
            kh_start = kh
            kh_offsets = kh_start + tl.arange(0, BLOCK_K)
            kh_mask = kh_offsets < K_h
            
            for kw in range(0, K_w, BLOCK_K):
                kw_start = kw
                kw_offsets = kw_start + tl.arange(0, BLOCK_K)
                kw_mask = kw_offsets < K_w
                
                # Compute input positions for this kernel position
                h_in = h_start * stride_h - pad_h + kh_offsets[None, None, :] * dil_h
                w_in = w_start * stride_w - pad_w + kw_offsets[None, :, None] * dil_w
                
                # Check bounds for input positions
                h_valid = (h_in >= 0) & (h_in < H)
                w_valid = (w_in >= 0) & (w_in < W)
                valid_mask = h_valid & w_valid
                
                # Compute linear offsets for input tensor
                # x_ptr shape: [B, C_in, H, W]
                # We need to gather values for all combinations of:
                # batch, c_in, h_in, w_in
                
                # Broadcast for shape compatibility
                batch_idx = pid_batch
                c_in_idx = c_in_offsets[:, None, None, None]
                h_in_idx = h_in[None, None, :, :]
                w_in_idx = w_in[None, None, :, :]
                
                # Create mask for valid indices
                c_in_idx_mask = c_in_idx < C_in
                h_in_idx_mask = h_in_idx < H
                w_in_idx_mask = w_in_idx < W
                idx_mask = c_in_idx_mask & h_in_idx_mask & w_in_idx_mask
                
                # Compute offsets for input
                # x_ptr + batch_offset + c_in_offset + h_offset + w_offset
                # where offsets are calculated for 4D tensor
                batch_offset = batch_idx * (C_in * H * W)
                c_in_offset = c_in_idx * (H * W)
                h_offset = h_in_idx * W
                w_offset = w_in_idx
                
                offsets_x = batch_offset + c_in_offset + h_offset + w_offset
                
                # Load input values with masks
                x_val = tl.load(
                    x_ptr + offsets_x,
                    mask=idx_mask & valid_mask[None, None, :, :],
                    other=0.0
                )
                
                # Load weights
                # w_ptr shape: [C_out, C_in, K_h, K_w]
                c_out_idx = c_out_offsets[:, None, None, None]
                kh_idx = kh_offsets[None, None, :, None]
                kw_idx = kw_offsets[None, None, None, :]
                
                w_offsets_full = (
                    c_out_idx * (C_in * K_h * K_w) +
                    c_in_idx * (K_h * K_w) +
                    kh_idx * K_w +
                    kw_idx
                )
                
                w_mask_full = (
                    (c_out_idx < C_out) & 
                    c_in_idx_mask & 
                    kh_mask[None, None, :, None] & 
                    kw_mask[None, None, None, :]
                )
                
                w_val = tl.load(
                    w_ptr + w_offsets_full,
                    mask=w_mask_full,
                    other=0.0
                )
                
                # Compute multiplication with broadcasting
                # x_val: [BLOCK_C_in, BLOCK_K, BLOCK_K] -> needs to be [BLOCK_C_in, BLOCK_H, BLOCK_W]
                # But BLOCK_H and BLOCK_W are 1 in this innermost loop
                # So we need to accumulate across kernel positions
                
                # For simplicity in this kernel, we'll handle the full convolution differently
                # Let's use a more efficient approach for the convolution
                
                # Reshape to enable matrix multiplication when possible
                # For now, do element-wise multiply and accumulate
                
                # Expand x_val for broadcasting: [BLOCK_C_in, BLOCK_H, BLOCK_W]
                # We need to broadcast x_val over kernel positions
                x_val_expanded = tl.broadcast_to(
                    x_val[:, 0, 0, :],
                    (BLOCK_C_in, BLOCK_H, BLOCK_W)
                )
                
                # Expand w_val: [BLOCK_C_out, BLOCK_C_in, BLOCK_K, BLOCK_K]
                # We need to broadcast across H and W positions
                w_val_expanded = tl.broadcast_to(
                    w_val[:, :, 0, 0],
                    (BLOCK_C_out, BLOCK_C_in, BLOCK_H, BLOCK_W)
                )
                
                # Compute contribution
                contrib = tl.sum(x_val_expanded * w_val_expanded, axis=1)
                acc += contrib
                
                # Handle remaining kernel elements if BLOCK_K < K_h or K_w
                if BLOCK_K < K_h:
                    for kh_idx in range(kh + BLOCK_K, min(K_h, kh + 2*BLOCK_K)):
                        kh_local = kh_idx - kh_start
                        h_in_local = h_start * stride_h - pad_h + kh_idx * dil_h
                        h_valid_local = (h_in_local >= 0) & (h_in_local < H)
                        
                        if h_valid_local:
                            for kw_idx in range(kw + BLOCK_K, min(K_w, kw + 2*BLOCK_K)):
                                kw_local = kw_idx - kw_start
                                w_in_local = w_start * stride_w - pad_w + kw_idx * dil_w
                                w_valid_local = (w_in_local >= 0) & (w_in_local < W)
                                
                                if w_valid_local:
                                    # Load input value at this position
                                    offsets_x_local = (
                                        batch_offset +
                                        c_in_offsets[:, None, None] * (H * W) +
                                        h_in_local * W +
                                        w_in_local
                                    )
                                    x_val_local = tl.load(
                                        x_ptr + offsets_x_local,
                                        mask=c_in_mask[:, None, None],
                                        other=0.0
                                    )
                                    
                                    # Load weight
                                    offsets_w_local = (
                                        c_out_offsets[:, None, None] * (C_in * K_h * K_w) +
                                        c_in_offsets[None, :, None] * (K_h * K_w) +
                                        kh_local * K_w +
                                        kw_local
                                    )
                                    w_val_local = tl.load(
                                        w_ptr + offsets_w_local,
                                        mask=c_out_mask[:, None, None] & c_in_mask[None, :, None],
                                        other=0.0
                                    )
                                    
                                    # Accumulate
                                    acc += tl.sum(x_val_local * w_val_local, axis=1)
                
                # Similar for remaining W positions if needed
                if BLOCK_K < K_w:
                    for kw_idx in range(kw + BLOCK_K, min(K_w, kw + 2*BLOCK_K)):
                        kw_local = kw_idx - kw_start
                        w_in_local = w_start * stride_w - pad_w + kw_idx * dil_w
                        w_valid_local = (w_in_local >= 0) & (w_in_local < W)
                        
                        if w_valid_local:
                            for kh_idx in range(kh, min(K_h, kh + BLOCK_K)):
                                kh_local = kh_idx - kh_start
                                h_in_local = h_start * stride_h - pad_h + kh_idx * dil_h
                                h_valid_local = (h_in_local >= 0) & (h_in_local < H)
                                
                                if h_valid_local:
                                    offsets_x_local = (
                                        batch_offset +
                                        c_in_offsets[:, None, None] * (H * W) +
                                        h_in_local * W +
                                        w_in_local
                                    )
                                    x_val_local = tl.load(
                                        x_ptr + offsets_x_local,
                                        mask=c_in_mask[:, None, None],
                                        other=0.0
                                    )
                                    
                                    offsets_w_local = (
                                        c_out_offsets[:, None, None] * (C_in * K_h * K_w) +
                                        c_in_offsets[None, :, None] * (K_h * K_w) +
                                        kh_local * K_w +
                                        kw_local
                                    )
                                    w_val_local = tl.load(
                                        w_ptr + offsets_w_local,
                                        mask=c_out_mask[:, None, None] & c_in_mask[None, :, None],
                                        other=0.0
                                    )
                                    
                                    acc += tl.sum(x_val_local * w_val_local, axis=1)
    
    # Add bias if present
    if HAS_BIAS:
        b_offsets = c_out_offsets
        b_mask = c_out_mask
        bias = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        bias_expanded = bias[:, None, None]
        acc += bias_expanded
    
    # Store output
    # y_ptr shape: [B, C_out, H_out, W_out]
    batch_offset_y = pid_batch * (C_out * H_out * W_out)
    c_out_offset_y = c_out_offsets[:, None, None] * (H_out * W_out)
    h_offset_y = h_offsets[None, :, None] * W_out
    w_offset_y = w_offsets[None, None, :]
    
    offsets_y = batch_offset_y + c_out_offset_y + h_offset_y + w_offset_y
    
    y_mask = (
        c_out_mask[:, None, None] & 
        h_mask[None, :, None] & 
        w_mask[None, None, :]
    )
    
    tl.store(y_ptr + offsets_y, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def triton_conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 2D convolution.
    Supports FP32 only as requested.
    """
    assert x.dtype == torch.float32, "Only FP32 is supported"
    assert weight.dtype == torch.float32, "Only FP32 is supported"
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Compute output dimensions
    H_out = (H + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    y = torch.empty((B, C_out, H_out, W_out), dtype=torch.float32, device=x.device)
    
    # Set block sizes (tunable for performance)
    BLOCK_C_out = 32
    BLOCK_C_in = 32
    BLOCK_K = 3
    BLOCK_H = 8
    BLOCK_W = 32
    
    # Grid dimensions
    grid = (
        B,  # batch
        triton.cdiv(C_out, BLOCK_C_out),  # output channels
        triton.cdiv(H_out, BLOCK_H),  # height
        triton.cdiv(W_out, BLOCK_W),  # width
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, y,
        B, C_in, H, W,
        C_out, K_h, K_w,
        stride, stride,
        padding, padding,
        dilation, dilation,
        H_out, W_out,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_K=BLOCK_K,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        HAS_BIAS=bias is not None,
        USE_MATRIX_MULTIPLY=False,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same weights as the original Conv2d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights similar to nn.Conv2d"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using our optimized Triton kernel.
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )