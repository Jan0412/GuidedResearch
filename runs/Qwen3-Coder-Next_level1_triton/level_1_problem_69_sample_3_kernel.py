import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to inputs and outputs
    X_ptr,  # Input: (batch, in_channels, H_in, W_in)
    W_ptr,  # Weight: (in_channels, out_channels, kH, kW)
    B_ptr,  # Bias: (out_channels,) or None
    Y_ptr,  # Output: (batch, out_channels, H_out, W_out)
    # Dimensions
    batch_size, in_channels, out_channels,
    H_in, W_in, H_out, W_out,
    kH, kW,
    stride_h, stride_w,
    padding_h, padding_w,
    output_padding_h, output_padding_w,
    dilation_h, dilation_w,
    # Block sizes for tiling
    BLOCK_C_in: tl.constexpr,
    BLOCK_C_out: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program IDs for output tensor dimensions
    pid_b = tl.program_id(0)  # batch
    pid_c_out = tl.program_id(1)  # output channel block
    pid_h = tl.program_id(2)  # output height block
    pid_w = tl.program_id(3)  # output width block
    
    # Calculate the starting positions in output tensor
    c_out_start = pid_c_out * BLOCK_C_out
    h_out_start = pid_h * BLOCK_H
    w_out_start = pid_w * BLOCK_W
    
    # Create output offsets
    offsets_c_out = c_out_start + tl.arange(0, BLOCK_C_out)
    offsets_h = h_out_start + tl.arange(0, BLOCK_H)
    offsets_w = w_out_start + tl.arange(0, BLOCK_W)
    
    # Create meshgrid for output spatial positions
    h_out_offsets = offsets_h[:, None] * W_out + offsets_w[None, :]
    w_out_offsets = offsets_w[None, :]
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_C_out, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(0, in_channels, BLOCK_C_in):
        offsets_c_in = c_in + tl.arange(0, BLOCK_C_in)
        
        # Load input data: X[batch, c_in, h_in, w_in]
        # Need to compute h_in = h_out * stride_h - padding_h + kH - 1 - dilation_h * (kH - 1 - i_h)
        # More precisely: h_in = h_out * stride_h - padding_h + i_h * dilation_h
        h_in_offsets = (h_out_offsets[:, :, None] * stride_h - padding_h + 
                       tl.arange(0, kH)[None, None, :] * dilation_h) * W_in
        w_in_offsets = w_out_offsets[:, :, None] + tl.arange(0, kW)[None, None, :] - padding_w
        
        # Create mask for valid input positions
        h_in_valid = (h_in_offsets >= 0) & (h_in_offsets < H_in * W_in)
        w_in_valid = (w_in_offsets >= 0) & (w_in_offsets < W_in)
        valid_mask = h_in_valid & w_in_valid
        
        # Compute actual indices
        h_in_flat = h_in_offsets * W_in + w_in_offsets
        h_in_flat = tl.where(valid_mask, h_in_flat, 0)
        
        # Load input values
        x_indices = pid_b * (in_channels * H_in * W_in) + \
                   offsets_c_in[:, None, None, None] * (H_in * W_in) + \
                   h_in_flat[None, :, :, :]
        
        # Reshape for proper indexing
        x_flat = tl.reshape(x_indices, (BLOCK_C_in, BLOCK_H * BLOCK_W * kH * kW))
        mask_x = (offsets_c_in[:, None, None, None] < in_channels) & valid_mask[None, :, :, :]
        mask_x = tl.reshape(mask_x, (BLOCK_C_in, BLOCK_H * BLOCK_W * kH * kW))
        
        # Load input - this is complex due to 4D indexing
        # Simplified approach: iterate through kernel positions
        for kh in range(kH):
            for kw in range(kW):
                # Compute input positions for this kernel position
                h_in = h_out_start * stride_h + kh * dilation_h - padding_h + tl.arange(0, BLOCK_H)[:, None] * stride_h
                w_in = w_out_start * stride_w + kw * dilation_w - padding_w + tl.arange(0, BLOCK_W)[None, :] * stride_w
                
                # Create masks for valid input positions
                mask_h = (h_in >= 0) & (h_in < H_in)
                mask_w = (w_in >= 0) & (w_in < W_in)
                mask = mask_h & mask_w
                
                # Compute linear indices
                h_in_clamped = tl.where(mask_h, h_in, 0)
                w_in_clamped = tl.where(mask_w, w_in, 0)
                idx = pid_b * (in_channels * H_in * W_in) + \
                      offsets_c_in[:, None, None] * (H_in * W_in) + \
                      h_in_clamped * W_in + w_in_clamped
                
                # Load input
                x_val = tl.load(X_ptr + idx, mask=mask[None, :, :], other=0.0)
                
                # Load corresponding weight
                w_idx = offsets_c_in[:, None, None] * (out_channels * kH * kW) + \
                        pid_c_out * (kH * kW) + \
                        kh * kW + kw
                w_val = tl.load(W_ptr + w_idx, mask=offsets_c_in[:, None, None] < in_channels, other=0.0)
                
                # Accumulate
                acc += x_val * tl.cast(w_val[None, :, :, :], tl.float32)
    
    # Add bias if present
    if B_ptr is not None:
        b_idx = pid_c_out * BLOCK_C_out + tl.arange(0, BLOCK_C_out)
        bias = tl.load(B_ptr + b_idx, mask=b_idx < out_channels, other=0.0)
        acc += bias[:, None, None]
    
    # Store output
    y_idx = pid_b * (out_channels * H_out * W_out) + \
            offsets_c_out[:, None, None] * (H_out * W_out) + \
            h_out_offsets[None, :, :] + w_out_offsets[None, :, :]
    
    acc = tl.cast(acc, tl.float32)
    mask_out = (offsets_c_out[:, None, None] < out_channels) & \
               (offsets_h[None, :, None] < H_out) & \
               (offsets_w[None, None, :] < W_out)
    
    tl.store(Y_ptr + y_idx, acc, mask=mask_out)


def triton_conv_transpose2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), 
                           output_padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of ConvTranspose2d forward pass.
    Assumes groups=1 for simplicity (can be extended for grouped convolutions).
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, H_in, W_in = x.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kH - 1) + output_padding[0] + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kW - 1) + output_padding[1] + 1
    
    # Create output tensor
    y = torch.empty(batch_size, out_channels, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes for tiling
    BLOCK_C_in = min(32, in_channels)
    BLOCK_C_out = min(32, out_channels)
    BLOCK_H = 8
    BLOCK_W = 8
    
    # Calculate grid dimensions
    grid = lambda meta: (
        batch_size,
        triton.cdiv(out_channels, meta['BLOCK_C_out']),
        triton.cdiv(H_out, meta['BLOCK_H']),
        triton.cdiv(W_out, meta['BLOCK_W'])
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels,
        H_in, W_in, H_out, W_out,
        kH, kW,
        stride[0], stride[1],
        padding[0], padding[1],
        output_padding[0], output_padding[1],
        dilation[0], dilation[1],
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, 
            dilation=self.dilation, 
            groups=self.groups
        )