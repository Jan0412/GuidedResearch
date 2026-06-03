import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # (batch_size, in_channels, H, W)
    w_ptr,  # (out_channels, in_channels // groups, kernel_h, kernel_w)
    b_ptr,  # (out_channels,) or None
    out_ptr,  # (batch_size, out_channels, H_out, W_out)
    # Shapes
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    # Derived dimensions
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    # Block sizes
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_CH: tl.constexpr,
    BLOCK_IN_CH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute batch index
    batch_idx = pid_b * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    mask_b = batch_idx < batch_size
    
    # Compute output channel indices
    out_c_start = pid_out_c * BLOCK_OUT_CH
    out_c_indices = out_c_start + tl.arange(0, BLOCK_OUT_CH)
    mask_out_c = out_c_indices < out_channels
    
    # Compute output spatial positions
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    h_indices = h_start + tl.arange(0, BLOCK_H)[:, None]
    w_indices = w_start + tl.arange(0, BLOCK_W)[None, :]
    
    # Compute input spatial positions (with padding)
    h_in = h_indices * stride_h - pad_h + tl.arange(0, kernel_h)[None, :, None] * dilation_h
    w_in = w_indices * stride_w - pad_w + tl.arange(0, kernel_w)[None, None, :] * dilation_w
    
    # Create masks for valid input positions
    mask_h = (h_in >= 0) & (h_in < height)
    mask_w = (w_in >= 0) & (w_in < width)
    mask_hw = mask_h & mask_w
    
    # Compute input channel indices per group
    in_ch_per_group = in_channels // groups
    out_ch_per_group = out_channels // groups
    
    group_id = pid_out_c // (out_ch_per_group)
    in_ch_start = group_id * in_ch_per_group
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_BATCH, BLOCK_OUT_CH, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over input channels in blocks
    for ch_block in range(0, in_ch_per_group, BLOCK_IN_CH):
        in_c_indices = in_ch_start + ch_block + tl.arange(0, BLOCK_IN_CH)
        mask_in_c = in_c_indices < (in_ch_start + in_ch_per_group)
        
        # Reshape masks for broadcasting
        mask_in_c_vec = mask_in_c[None, :, None, None]  # (1, BLOCK_IN_CH, 1, 1)
        
        # Load input: (batch, in_c, h_in, w_in)
        x_val = tl.load(
            x_ptr + batch_idx[:, None, None, None] * (in_channels * height * width) +
            in_c_indices[None, :, None, None] * (height * width) +
            h_in[None, None, :, :] * width +
            w_in[None, None, :, :],
            mask=mask_b[:, None, None, None] & mask_in_c_vec & mask_hw[None, None, :, :],
            other=0.0
        )
        
        # Load weights: (out_c, in_c, kh, kw)
        w_val = tl.load(
            w_ptr + (out_c_indices[:, None, None, None] * (in_channels * kernel_h * kernel_w) +
                    in_c_indices[None, :, None, None] * (kernel_h * kernel_w) +
                    tl.arange(0, kernel_h)[None, None, :, None] * kernel_w +
                    tl.arange(0, kernel_w)[None, None, None, :]),
            mask=mask_out_c[:, None, None, None] & mask_in_c_vec,
            other=0.0
        )
        
        # Compute partial convolution: (batch, out_c, h, w)
        # We need to broadcast and sum over in_c dimension
        # x_val: (B, in_c, kh, kw)
        # w_val: (out_c, in_c, kh, kw)
        # result: (B, out_c, h, w)
        partial = tl.sum(x_val[:, None, :, :, :] * w_val[None, :, :, :, :], axis=2)  # sum over kh
        partial = tl.sum(partial[:, :, None, :, :] * tl.broadcast_to(
            tl.arange(0, 1)[None, None, :, None], partial.shape
        ) * x_val[:, None, :, :, :], axis=2)  # This approach is inefficient; let's do proper matmul-like
        
        # Better approach: reshape to 2D matmul
        # We'll do it differently for efficiency
    
    # Alternative implementation: use a more efficient approach by reshaping to matrix multiplication
    # But for simplicity and correctness, let's implement the direct convolution with tiling
    
    # Re-implementing using a more efficient approach:
    # For each output position (b, oc, oh, ow), compute sum over ic, kh, kw
    
    # Since Triton doesn't easily support nested loops over multiple dimensions efficiently,
    # we'll implement the standard loop-based convolution kernel
    
    # Compute output position
    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    
    # Compute input position range
    ih0 = oh * stride_h - pad_h
    iw0 = ow * stride_w - pad_w
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_BATCH, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Compute input channel range for this group
        ic_start = g * in_ch_per_group
        ic_end = ic_start + in_ch_per_group
        ic_indices = tl.arange(0, BLOCK_IN_CH)
        
        # Process input channels in blocks
        for ic_offset in range(0, in_ch_per_group, BLOCK_IN_CH):
            ic_block = ic_start + ic_offset + ic_indices
            mask_ic = ic_block < ic_end
            
            # Compute input values for all batch, ih, iw, ic_block
            # ih = ih0 + kh * dilation_h
            # iw = iw0 + kw * dilation_w
            
            # Reshape for broadcasting
            mask_ic_vec = mask_ic[None, :, None, None]  # (1, BLOCK_IN_CH, 1, 1)
            
            # Load input: shape (B, ic_block, kh, kw)
            x_block = tl.load(
                x_ptr + batch_idx[:, None, None, None] * (in_channels * height * width) +
                (ic_block[None, :, None, None]) * (height * width) +
                (ih0[:, None, :, None] + tl.arange(0, kernel_h)[None, None, :, None] * dilation_h)[None, None, :, :] * width +
                (iw0[None, None, :, :] + tl.arange(0, kernel_w)[None, None, None, :] * dilation_w)[None, None, :, :],
                mask=mask_b[:, None, None, None] & mask_ic_vec &
                    ((ih0[:, None, :, None] + tl.arange(0, kernel_h)[None, None, :, None] * dilation_h) < height)[None, None, :, :] &
                    ((iw0[None, None, :, :] + tl.arange(0, kernel_w)[None, None, None, :] * dilation_w) < width)[None, None, :, :],
                other=0.0
            )
            
            # Load weights: shape (out_c_block, ic_block, kh, kw)
            out_c_block_start = pid_out_c * BLOCK_OUT_CH
            out_c_block = out_c_block_start + tl.arange(0, BLOCK_OUT_CH)
            mask_out_c_block = out_c_block < out_channels
            
            w_block = tl.load(
                w_ptr + (out_c_block[:, None, None, None] * (in_channels * kernel_h * kernel_w) +
                        (ic_block[None, :, None, None]) * (kernel_h * kernel_w) +
                        tl.arange(0, kernel_h)[None, None, :, None] * kernel_w +
                        tl.arange(0, kernel_w)[None, None, None, :]),
                mask=mask_out_c_block[:, None, None, None] & mask_ic_vec,
                other=0.0
            )
            
            # Compute convolution for this block
            # x_block: (B, ic_block, kh, kw)
            # w_block: (out_c_block, ic_block, kh, kw)
            # result: (B, out_c_block, h, w)
            
            # Reshape to 2D matrix multiply: (B * h * w, ic_block) @ (ic_block, out_c_block)
            x_flat = x_block.permute(0, 2, 3, 1).reshape(BLOCK_BATCH * BLOCK_H * BLOCK_W, BLOCK_IN_CH)
            w_flat = w_block.permute(1, 0, 2, 3).reshape(BLOCK_IN_CH, BLOCK_OUT_CH)
            partial = tl.dot(x_flat, w_flat)
            partial = partial.reshape(BLOCK_BATCH, BLOCK_H, BLOCK_W, BLOCK_OUT_CH)
            partial = partial.permute(0, 3, 1, 2)  # (B, out_c_block, h, w)
            
            acc += partial
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_indices, mask=mask_out_c)
        acc += bias[None, :, None, None]
    
    # Store output
    tl.store(
        out_ptr + batch_idx[:, None, None] * (out_channels * out_h * out_w) +
        out_c_indices[None, :, None] * (out_h * out_w) +
        (pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None])[None, None, :] * out_w +
        (pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :])[None, None, :],
        acc,
        mask=mask_b[:, None, None] & mask_out_c[None, :, None] &
             ((pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]) < out_h)[None, None, :] &
             ((pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]) < out_w)[None, None, :]
    )


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Triton-based 2D convolution with support for asymmetric kernels and arbitrary parameters.
    """
    # Ensure inputs are contiguous and on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Convert stride, padding, dilation to tuples if they're not already
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    
    # Calculate output dimensions
    out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Allocate output tensor
    out = torch.empty((batch_size, out_channels, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Set block sizes (tunable parameters)
    BLOCK_BATCH = 1
    BLOCK_OUT_CH = 16
    BLOCK_IN_CH = 32
    BLOCK_H = 8
    BLOCK_W = 8
    
    # Calculate grid dimensions
    grid = (
        (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH,
        (out_channels + BLOCK_OUT_CH - 1) // BLOCK_OUT_CH,
        (out_h + BLOCK_H - 1) // BLOCK_H,
        (out_w + BLOCK_W - 1) // BLOCK_W
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups,
        out_h, out_w,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT_CH=BLOCK_OUT_CH,
        BLOCK_IN_CH=BLOCK_IN_CH,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replace PyTorch's native conv2d with our Triton implementation
        # Ensure the weights and bias are on the same device as input
        weight = self.conv2d.weight
        bias = self.conv2d.bias if self.conv2d.bias is not None else None
        
        return triton_conv2d(
            x, weight, bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups
        )