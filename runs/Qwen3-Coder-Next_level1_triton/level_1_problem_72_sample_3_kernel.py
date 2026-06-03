import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to input tensors
    x_ptr,  # [B, C_in, D, H, W]
    w_ptr,  # [C_in, C_out // G, Kd, Kh, Kw] (transposed conv weights)
    # Pointers to output tensor
    out_ptr,  # [B, C_out, D_out, H_out, W_out]
    # Optional bias
    b_ptr,  # [C_out] or None
    # Dimensions
    B, C_in, C_out, G,  # batch, channels, groups
    D_in, H_in, W_in,   # input dimensions
    D_out, H_out, W_out,  # output dimensions
    Kd, Kh, Kw,         # kernel dimensions
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    # Block sizes
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program indices
    pid_b = tl.program_id(0)  # batch index
    pid_c_out = tl.program_id(1)  # output channel block index
    pid_d = tl.program_id(2)  # depth position
    pid_h = tl.program_id(3)  # height position
    pid_w = tl.program_id(4)  # width position

    # Calculate output channel range
    c_out_offsets = pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_offsets < C_out

    # Calculate input positions that contribute to this output position
    # For transposed convolution: output_pos = input_pos * stride + (kernel_pos - 1) - pad + output_pad
    # So input_pos = (output_pos - (kernel_pos - 1) + pad - output_pad) // stride
    
    # Loop over kernel positions
    total_output = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)
    
    # Calculate base input position for this output position
    base_d = pid_d * stride_d - pad_d + output_pad_d
    base_h = pid_h * stride_h - pad_h + output_pad_w
    base_w = pid_w * stride_w - pad_w + output_pad_w

    # Loop over kernel dimensions
    for kd in range(Kd):
        input_d = base_d + kd
        d_valid = (input_d >= 0) & (input_d < D_in)
        
        for kh in range(Kh):
            input_h = base_h + kh
            h_valid = (input_h >= 0) & (input_h < H_in)
            
            for kw in range(Kw):
                input_w = base_w + kw
                w_valid = (input_w >= 0) & (input_w < W_in)
                
                # Check if this kernel position contributes to valid output
                valid = d_valid & h_valid & w_valid
                
                # Load kernel weights: shape [C_in, C_out // G, Kd, Kh, Kw]
                # For group convolution: C_in_group = C_in // G, C_out_group = C_out // G
                # We need to access w[c_in, c_out // G, kd, kh, kw]
                
                # Load input: x[pid_b, c_in, input_d, input_h, input_w]
                # For grouped convolution: we need to ensure c_in and c_out are in same group
                
                # Calculate group indices
                c_out_group = pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
                c_in_group = tl.arange(0, BLOCK_SIZE_C_IN)
                
                # Loop over input channels in blocks
                for pid_c_in in range((C_in + BLOCK_SIZE_C_IN - 1) // BLOCK_SIZE_C_IN):
                    c_in_offsets = pid_c_in * BLOCK_SIZE_C_IN + tl.arange(0, BLOCK_SIZE_C_IN)
                    c_in_mask = c_in_offsets < C_in
                    
                    # Calculate group info
                    group_idx = c_in_offsets // (C_in // G)
                    c_in_in_group = c_in_offsets % (C_in // G)
                    
                    # Only process if c_out and c_in are in the same group (for groups > 1)
                    if G > 1:
                        c_out_group_idx = c_out_group // (C_out // G)
                        mask = (group_idx == c_out_group_idx) & c_in_mask & c_out_mask & valid
                    else:
                        mask = c_in_mask & c_out_mask & valid
                    
                    if not tl.any(mask):
                        continue
                    
                    # Load input: x[pid_b, c_in, input_d, input_h, input_w]
                    x_offset = pid_b * (C_in * D_in * H_in * W_in) + \
                              c_in_offsets[:, None] * (D_in * H_in * W_in) + \
                              input_d * (H_in * W_in) + \
                              input_h * W_in + \
                              input_w
                    x_val = tl.load(x_ptr + x_offset, mask=c_in_mask[:, None], other=0.0)
                    
                    # Load weights: w[c_in, c_out // G, kd, kh, kw]
                    # Weight layout: [C_in, C_out // G, Kd, Kh, Kw]
                    w_offset = c_in_offsets[:, None] * (C_out // G * Kd * Kh * Kw) + \
                              (c_out_group[None, :] // (C_out // G)) * (Kd * Kh * Kw) + \
                              kd * (Kh * Kw) + \
                              kh * Kw + \
                              kw
                    w_val = tl.load(w_ptr + w_offset, mask=mask, other=0.0)
                    
                    # Accumulate: out[c_out] += x[c_in] * w[c_in, c_out // G, ...]
                    total_output += tl.sum(x_val * w_val, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        bias_offsets = c_out_offsets
        bias_val = tl.load(b_ptr + bias_offsets, mask=c_out_mask, other=0.0)
        total_output += bias_val
    
    # Store result
    out_offset = pid_b * (C_out * D_out * H_out * W_out) + \
                c_out_offsets * (D_out * H_out * W_out) + \
                pid_d * (H_out * W_out) + \
                pid_h * W_out + \
                pid_w
    tl.store(out_ptr + out_offset, total_output, mask=c_out_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    output_padding=(0, 0, 0),
    groups=1
) -> torch.Tensor:
    """
    Custom Triton implementation of 3D transposed convolution.
    
    Args:
        x: Input tensor of shape [B, C_in, D, H, W]
        weight: Weight tensor of shape [C_in, C_out // G, Kd, Kh, Kw]
        bias: Optional bias tensor of shape [C_out]
        stride, padding, output_padding, groups: convolution parameters
    
    Returns:
        Output tensor of shape [B, C_out, D_out, H_out, W_out]
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D_in, H_in, W_in = x.shape
    # Weight shape for ConvTranspose3d is [C_in, C_out // G, Kd, Kh, Kw]
    C_out = weight.shape[1] * groups
    
    # Calculate output dimensions
    Kd, Kh, Kw = weight.shape[2], weight.shape[3], weight.shape[4]
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + Kd + output_padding[0]
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + Kh + output_padding[1]
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + Kw + output_padding[2]
    
    # Prepare output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Configure grid and block sizes
    # Grid: [batch, output_channel_blocks, depth, height, width]
    BLOCK_SIZE_C_OUT = 16  # Tunable parameter
    BLOCK_SIZE_C_IN = 8    # Tunable parameter
    BLOCK_SIZE_K = 4       # Tunable parameter
    
    grid = (
        B,
        (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT,
        D_out,
        H_out,
        W_out
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, out, bias,
        B, C_in, C_out, groups,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        Kd, Kh, Kw,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the 3D transposed convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weight and bias (same initialization as PyTorch)
        # For transposed conv, weight shape is [in_channels, out_channels // groups, *kernel_size]
        fan_in = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2] // groups
        std = 1.0 / (fan_in ** 0.5)
        
        # Create weight parameter
        weight_shape = [in_channels, out_channels // groups] + list(kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape) * std)
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels) * std)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution using Triton.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )