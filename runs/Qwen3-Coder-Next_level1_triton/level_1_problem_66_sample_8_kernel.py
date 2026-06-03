import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_out, C_in, Kd, Kh, Kw)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, D, H, W,
    C_out, Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    # Output tensor dimensions
    D_out, H_out, W_out,
    # Block sizes for tiling
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_Kd: tl.constexpr,
    BLOCK_Kh: tl.constexpr,
    BLOCK_Kw: tl.constexpr,
    BLOCK_D_out: tl.constexpr,
    BLOCK_H_out: tl.constexpr,
    BLOCK_W_out: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_c_out = tl.program_id(1)  # output channel index
    pid_d_out = tl.program_id(2)  # depth output index
    pid_h_out = tl.program_id(3)  # height output index
    pid_w_out = tl.program_id(4)  # width output index
    
    # Calculate actual output coordinates
    d_out = pid_d_out * BLOCK_D_out + tl.arange(0, BLOCK_D_out)
    h_out = pid_h_out * BLOCK_H_out + tl.arange(0, BLOCK_H_out)
    w_out = pid_w_out * BLOCK_W_out + tl.arange(0, BLOCK_W_out)
    
    # Create meshgrid for output coordinates
    d_out_grid, h_out_grid, w_out_grid = tl.meshgrid(d_out, h_out, w_out, indexing='ij')
    
    # Calculate input coordinates (with padding and stride)
    d_in = d_out_grid * stride_d - padding_d + (tl.arange(0, Kd)[:, None, None] * dilation_d)
    h_in = h_out_grid * stride_h - padding_h + (tl.arange(0, Kh)[None, :, None] * dilation_h)
    w_in = w_out_grid * stride_w - padding_w + (tl.arange(0, Kw)[None, None, :] * dilation_w)
    
    # Check if input coordinates are valid
    valid_mask = (
        (d_in >= 0) & (d_in < D) &
        (h_in >= 0) & (h_in < H) &
        (w_in >= 0) & (w_in < W)
    )
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_D_out, BLOCK_H_out, BLOCK_W_out), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for c_in_offset in range(0, C_in, BLOCK_C_in):
        c_in_idx = c_in_offset + tl.arange(0, BLOCK_C_in)
        
        # Loop over output channels in blocks
        for c_out_offset in range(0, BLOCK_C_out):
            c_out_idx = pid_c_out * BLOCK_C_out + c_out_offset
            
            # Load input block
            # Calculate input pointer offset
            # Shape: (B, C_in, D, H, W) -> flattened index = b*C_in*D*H*W + c_in*D*H*W + d*H*W + h*W + w
            input_offsets = (
                pid_b * C_in * D * H * W +
                c_in_idx[:, None, None, None] * D * H * W +
                d_in[None, :, :, :] * H * W +
                h_in[None, :, :, :] * W +
                w_in[None, :, :, :]
            )
            
            # Reshape for memory access
            input_offsets = tl.reshape(input_offsets, (BLOCK_C_in, BLOCK_D_out * BLOCK_H_out * BLOCK_W_out))
            valid_mask_flat = tl.reshape(valid_mask, (BLOCK_C_in, BLOCK_D_out * BLOCK_H_out * BLOCK_W_out))
            
            # Load input data
            x_block = tl.load(
                x_ptr + input_offsets,
                mask=valid_mask_flat,
                other=0.0
            )
            
            # Load weight block
            # Weight shape: (C_out, C_in, Kd, Kh, Kw)
            weight_offsets = (
                c_out_idx * C_in * Kd * Kh * Kw +
                c_in_idx[:, None, None, None] * Kd * Kh * Kw +
                tl.arange(0, Kd)[:, None, None, None] * Kh * Kw +
                tl.arange(0, Kh)[None, :, None, None] * Kw +
                tl.arange(0, Kw)[None, None, :, None]
            )
            
            weight_offsets = tl.reshape(weight_offsets, (BLOCK_C_in, Kd * Kh * Kw))
            w_block = tl.load(w_ptr + weight_offsets)
            
            # Reshape weight for computation
            w_block = tl.reshape(w_block, (BLOCK_C_in, Kd, Kh, Kw))
            
            # Compute convolution
            # Expand dimensions for broadcasting
            x_block_expanded = x_block[:, :, None, None, None]  # (C_in, D_out*H_out*W_out, 1, 1, 1)
            w_block_expanded = w_block[:, None, :, :, :]  # (C_in, 1, Kd, Kh, Kw)
            
            # Reshape x_block for element-wise multiplication with weights
            x_reshaped = tl.reshape(x_block, (BLOCK_C_in, BLOCK_D_out, BLOCK_H_out, BLOCK_W_out))
            
            # Multiply and accumulate
            for kd in range(Kd):
                for kh in range(Kh):
                    for kw in range(Kw):
                        # Get valid mask for this kernel position
                        kernel_valid = valid_mask & (d_in == d_out_grid * stride_d - padding_d + kd * dilation_d) & \
                                      (h_in == h_out_grid * stride_h - padding_h + kh * dilation_h) & \
                                      (w_in == w_out_grid * stride_w - padding_w + kw * dilation_w)
                        
                        # Only compute for valid positions
                        if tl.sum(kernel_valid) > 0:
                            # Get input values for this kernel position
                            x_kernel = tl.where(
                                kernel_valid[None, :, :, :],
                                x_reshaped,
                                0.0
                            )
                            
                            # Get weight value
                            w_val = tl.load(w_ptr + c_out_idx * C_in * Kd * Kh * Kw + 
                                          c_in_idx * Kd * Kh * Kw + 
                                          kd * Kh * Kw + kh * Kw + kw)
                            
                            # Multiply and accumulate
                            acc += tl.sum(x_kernel * w_val[None, :, :, :], axis=0)
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out * BLOCK_C_out + tl.arange(0, BLOCK_C_out))
        acc += bias[None, :, :, :]
    
    # Store output
    # Calculate output pointer offset
    output_offsets = (
        pid_b * C_out * D_out * H_out * W_out +
        (pid_c_out * BLOCK_C_out + tl.arange(0, BLOCK_C_out)[:, None, None]) * D_out * H_out * W_out +
        d_out_grid * H_out * W_out +
        h_out_grid * W_out +
        w_out_grid
    )
    
    output_offsets = tl.reshape(output_offsets, (BLOCK_C_out, BLOCK_D_out * BLOCK_H_out * BLOCK_W_out))
    
    # Store results
    tl.store(
        out_ptr + output_offsets,
        tl.reshape(acc, (BLOCK_C_out, BLOCK_D_out * BLOCK_H_out * BLOCK_W_out)),
        mask=True  # We assume all output positions are valid
    )


def triton_conv3d(x, weight, bias, stride, padding, dilation, groups):
    """
    Triton implementation of 3D convolution
    """
    # Get dimensions
    B, C_in, D, H, W = x.shape
    C_out, C_in_w, Kd, Kh, Kw = weight.shape
    
    assert C_in == C_in_w, "Input channels must match"
    assert groups == 1, "Only groups=1 is supported in this implementation"
    
    # Calculate output dimensions
    D_out = (D + 2 * padding[0] - dilation[0] * (Kd - 1) - 1) // stride[0] + 1
    H_out = (H + 2 * padding[1] - dilation[1] * (Kh - 1) - 1) // stride[1] + 1
    W_out = (W + 2 * padding[2] - dilation[2] * (Kw - 1) - 1) // stride[2] + 1
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes for tiling
    BLOCK_C_out = min(32, C_out)
    BLOCK_C_in = min(16, C_in)
    BLOCK_D_out = min(4, D_out)
    BLOCK_H_out = min(8, H_out)
    BLOCK_W_out = min(8, W_out)
    BLOCK_Kd = Kd
    BLOCK_Kh = Kh
    BLOCK_Kw = Kw
    
    # Calculate grid dimensions
    grid = (
        B,  # batch
        (C_out + BLOCK_C_out - 1) // BLOCK_C_out,  # output channels
        (D_out + BLOCK_D_out - 1) // BLOCK_D_out,  # depth
        (H_out + BLOCK_H_out - 1) // BLOCK_H_out,  # height
        (W_out + BLOCK_W_out - 1) // BLOCK_W_out   # width
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, D, H, W,
        C_out, Kd, Kh, Kw,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        D_out, H_out, W_out,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_C_in=BLOCK_C_in,
        BLOCK_D_out=BLOCK_D_out,
        BLOCK_H_out=BLOCK_H_out,
        BLOCK_W_out=BLOCK_W_out,
        BLOCK_Kd=BLOCK_Kd,
        BLOCK_Kh=BLOCK_Kh,
        BLOCK_Kw=BLOCK_Kw,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D convolution model using custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        """
        return triton_conv3d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

# Import math for initialization
import math