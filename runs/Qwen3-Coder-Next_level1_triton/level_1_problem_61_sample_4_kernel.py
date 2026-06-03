import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    x_ptr,  # Input: (B, C_in, D, H, W)
    w_ptr,  # Weight: (C_in, C_out, Kd, Kh, Kw)
    b_ptr,  # Bias: (C_out,)
    out_ptr,  # Output: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, C_out,
    D, H, W,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    output_padding_d, output_padding_h, output_padding_w,
    pad_d, pad_h, pad_w,
    # Strides for input tensor
    stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    # Strides for weight tensor
    stride_w_ci, stride_w_co, stride_w_kd, stride_w_kh, stride_w_kw,
    # Strides for output tensor
    stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    # Block sizes
    BLOCK_CIN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_CDIM: tl.constexpr,
):
    # Program IDs for output tensor dimensions
    pid_b = tl.program_id(0)  # batch
    pid_c_out = tl.program_id(1)  # output channel
    pid_d = tl.program_id(2)  # output depth
    pid_h = tl.program_id(3)  # output height
    pid_w = tl.program_id(4)  # output width
    
    # Calculate the starting position in output
    out_d = pid_d * BLOCK_D
    out_h = pid_h * BLOCK_H
    out_w = pid_w * BLOCK_W
    
    # Accumulator for the result
    accumulator = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for off_c_in in range(0, C_in, BLOCK_CIN):
        c_in_start = off_c_in
        c_in_end = tl.minimum(c_in_start + BLOCK_CIN, C_in)
        
        # Loop over kernel dimensions
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Calculate input position corresponding to this kernel position
                    in_d = pid_d * stride_d + kd - pad_d + output_padding_d
                    in_h = pid_h * stride_h + kh - pad_h + output_padding_h
                    in_w = pid_w * stride_w + kw - pad_w + output_padding_w
                    
                    # Check if input position is valid
                    valid_input = (in_d >= 0) & (in_d < D) & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
                    
                    # Load input values if valid
                    if valid_input:
                        x_offsets = (
                            pid_b * stride_x_b +
                            c_in_start * stride_x_c +
                            in_d * stride_x_d +
                            in_h * stride_x_h +
                            in_w * stride_x_w
                        )
                        
                        # Create offsets for the block
                        d_offsets = tl.arange(0, BLOCK_D)
                        h_offsets = tl.arange(0, BLOCK_H)
                        w_offsets = tl.arange(0, BLOCK_W)
                        
                        d_mask = (out_d + d_offsets < pid_d * BLOCK_D + BLOCK_D) & (out_d + d_offsets < D)
                        h_mask = (out_h + h_offsets < pid_h * BLOCK_H + BLOCK_H) & (out_h + h_offsets < H)
                        w_mask = (out_w + w_offsets < pid_w * BLOCK_W + BLOCK_W) & (out_w + w_offsets < W)
                        
                        # Create 3D mask
                        d_idx = (out_d + d_offsets)[:, None, None]
                        h_idx = (out_h + h_offsets)[None, :, None]
                        w_idx = (out_w + w_offsets)[None, None, :]
                        
                        d_mask_3d = d_mask[:, None, None]
                        h_mask_3d = h_mask[None, :, None]
                        w_mask_3d = w_mask[None, None, :]
                        mask = d_mask_3d & h_mask_3d & w_mask_3d
                        
                        # Load input block
                        x_val = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
                        
                        # Load weight value for this kernel position
                        w_offset = (
                            (c_in_start + tl.arange(0, BLOCK_CIN))[:None, None, None] * stride_w_ci +
                            pid_c_out * stride_w_co +
                            kd * stride_w_kd +
                            kh * stride_w_kh +
                            kw * stride_w_kw
                        )
                        w_val = tl.load(w_ptr + w_offset, mask=mask, other=0.0)
                        
                        # Accumulate: x * w
                        accumulator += tl.sum(x_val * w_val, axis=0)
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        accumulator += bias
    
    # Store result
    out_offsets = (
        pid_b * stride_out_b +
        pid_c_out * stride_out_c +
        (out_d + tl.arange(0, BLOCK_D)[:, None, None]) * stride_out_d +
        (out_h + tl.arange(0, BLOCK_H)[None, :, None]) * stride_out_h +
        (out_w + tl.arange(0, BLOCK_W)[None, None, :]) * stride_out_w
    )
    
    d_mask_3d = (out_d + tl.arange(0, BLOCK_D)[:, None, None] < D) & (out_d + tl.arange(0, BLOCK_D)[:, None, None] >= 0)
    h_mask_3d = (out_h + tl.arange(0, BLOCK_H)[None, :, None] < H) & (out_h + tl.arange(0, BLOCK_H)[None, :, None] >= 0)
    w_mask_3d = (out_w + tl.arange(0, BLOCK_W)[None, None, :] < W) & (out_w + tl.arange(0, BLOCK_W)[None, None, :] >= 0)
    mask = d_mask_3d & h_mask_3d & w_mask_3d
    
    tl.store(out_ptr + out_offsets, accumulator.to(x_ptr.dtype.element_ty), mask=mask)


def triton_conv_transpose3d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of 3D transposed convolution.
    
    Args:
        x: Input tensor (B, C_in, D, H, W)
        weight: Weight tensor (C_in, C_out, Kd, Kh, Kw)
        bias: Bias tensor (C_out,) or None
        stride, padding, output_padding, groups: convolution parameters
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = x.shape
    C_in_w, C_out, Kd, Kh, Kw = weight.shape
    
    # Validate dimensions
    assert C_in == C_in_w, f"Input channels {C_in} doesn't match weight input channels {C_in_w}"
    assert groups == 1, "Only groups=1 is supported in this implementation"
    
    # Calculate output dimensions
    D_out = (D - 1) * stride - 2 * padding + output_padding + Kd
    H_out = (H - 1) * stride - 2 * padding + output_padding + Kh
    W_out = (W - 1) * stride - 2 * padding + output_padding + Kw
    
    # Create output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Configure kernel parameters
    stride_d, stride_h, stride_w = stride, stride, stride
    pad_d, pad_h, pad_w = padding, padding, padding
    output_padding_d, output_padding_h, output_padding_w = output_padding, output_padding, output_padding
    
    # Strides for input tensor
    stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w = x.stride()
    # Strides for weight tensor
    stride_w_ci, stride_w_co, stride_w_kd, stride_w_kh, stride_w_kw = weight.stride()
    # Strides for output tensor
    stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w = out.stride()
    
    # Grid configuration - one block per output element region
    # We'll use a reasonable block size for each dimension
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4
    BLOCK_CIN = 16  # Tile over input channels
    
    # Calculate grid dimensions
    grid_d = (D_out + BLOCK_D - 1) // BLOCK_D
    grid_h = (H_out + BLOCK_H - 1) // BLOCK_H
    grid_w = (W_out + BLOCK_W - 1) // BLOCK_W
    
    # Launch kernel
    conv_transpose3d_kernel[(
        B,  # batch
        C_out,  # output channels
        grid_d,  # depth blocks
        grid_h,  # height blocks
        grid_w   # width blocks
    )](
        x, weight, bias, out,
        B, C_in, C_out,
        D, H, W,
        Kd, Kh, Kw,
        stride_d, stride_h, stride_w,
        output_padding_d, output_padding_h, output_padding_w,
        pad_d, pad_h, pad_w,
        stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
        stride_w_ci, stride_w_co, stride_w_kd, stride_w_kh, stride_w_kw,
        stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
        BLOCK_CIN=BLOCK_CIN,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_CDIM=1
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed 3D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Create the weight and bias parameters
        # Note: nn.ConvTranspose3d weight shape is (in_channels, out_channels, Kd, Kh, Kw)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and bias similar to PyTorch's ConvTranspose3d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using our custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict handling the parameter name mapping."""
        # The original model uses 'conv_transpose3d.weight' and 'conv_transpose3d.bias'
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('conv_transpose3d.'):
                new_key = key.replace('conv_transpose3d.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return super().load_state_dict(new_state_dict, strict)


# Import math for initialization
import math