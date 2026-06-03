import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer (N, C, D, H, W)
    w_ptr,  # Weight tensor pointer (OutC, InC, Kd, Kh, Kw)
    b_ptr,  # Bias pointer (OutC,) or None
    out_ptr,  # Output tensor pointer (N, OutC, Od, Oh, Ow)
    # Dimensions
    N, C, D, H, W,  # Input dimensions
    OutC, Kd, Kh, Kw,  # Weight dimensions
    stride_d, stride_h, stride_w,  # Stride parameters
    pad_d, pad_h, pad_w,  # Padding parameters
    dil_d, dil_h, dil_w,  # Dilation parameters
    # Output dimensions
    Od, Oh, Ow,
    # Block sizes
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)  # batch index
    pid_outc = tl.program_id(1)  # output channel index
    pid_od = tl.program_id(2)  # output depth index
    pid_oh = tl.program_id(3)  # output height index
    pid_ow = tl.program_id(4)  # output width index
    
    # Compute output position
    out_d = pid_od * BLOCK_D
    out_h = pid_oh * BLOCK_H
    out_w = pid_ow * BLOCK_W
    
    # Calculate the starting input positions for this output position
    # For a given output position (od, oh, ow), the corresponding input position is:
    # id = od * stride_d - pad_d + kd * dil_d
    # ih = oh * stride_h - pad_h + kh * dil_h
    # iw = ow * stride_w - pad_w + kw * dil_w
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(C):
        # Loop over kernel depth
        for kd in range(Kd):
            id_start = out_d * stride_d - pad_d + kd * dil_d
            # Loop over kernel height
            for kh in range(Kh):
                ih_start = out_h * stride_h - pad_h + kh * dil_h
                # Loop over kernel width
                for kw in range(Kw):
                    iw_start = out_w * stride_w - pad_w + kw * dil_w
                    
                    # Load input values
                    # We need to handle bounds for the input
                    offsets_d = tl.arange(0, BLOCK_D)
                    offsets_h = tl.arange(0, BLOCK_H)
                    offsets_w = tl.arange(0, BLOCK_W)
                    
                    # Compute actual input positions
                    id_pos = id_start + offsets_d[:, None, None]
                    ih_pos = ih_start + offsets_h[None, :, None]
                    iw_pos = iw_start + offsets_w[None, None, :]
                    
                    # Create masks for valid input positions
                    mask_d = (id_pos >= 0) & (id_pos < D)
                    mask_h = (ih_pos >= 0) & (ih_pos < H)
                    mask_w = (iw_pos >= 0) & (iw_pos < W)
                    mask = mask_d & mask_h & mask_w
                    
                    # Calculate input indices
                    input_indices = (
                        pid_n * (C * D * H * W) +
                        c * (D * H * W) +
                        id_pos[:, :, None] * (H * W) +
                        ih_pos[:, None, :] * W +
                        iw_pos[None, :, :]
                    )
                    
                    # Reshape mask for proper broadcasting
                    mask_reshaped = mask[:, :, None] & mask_h[:, None, :] & mask_w[None, :, :]
                    mask_reshaped = mask_reshaped.reshape(BLOCK_D * BLOCK_H * BLOCK_W)
                    
                    # Load input values
                    input_vals = tl.load(
                        x_ptr + input_indices.reshape(BLOCK_D * BLOCK_H * BLOCK_W),
                        mask=mask_reshaped,
                        other=0.0
                    ).reshape(BLOCK_D, BLOCK_H, BLOCK_W)
                    
                    # Load weight value
                    weight_val = tl.load(
                        w_ptr + 
                        pid_outc * (C * Kd * Kh * Kw) +
                        c * (Kd * Kh * Kw) +
                        kd * (Kh * Kw) +
                        kh * Kw +
                        kw
                    )
                    
                    # Accumulate
                    acc += input_vals * weight_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_outc)
        acc += bias
    
    # Store results
    offsets_d = tl.arange(0, BLOCK_D)
    offsets_h = tl.arange(0, BLOCK_H)
    offsets_w = tl.arange(0, BLOCK_W)
    
    output_indices = (
        pid_n * (OutC * Od * Oh * Ow) +
        pid_outc * (Od * Oh * Ow) +
        (out_d + offsets_d[:, None, None]) * (Oh * Ow) +
        (out_h + offsets_h[None, :, None]) * Ow +
        (out_w + offsets_w[None, None, :])
    )
    
    # Create output mask
    mask_d_out = (out_d + offsets_d[:, None, None]) < Od
    mask_h_out = (out_h + offsets_h[None, :, None]) < Oh
    mask_w_out = (out_w + offsets_w[None, None, :]) < Ow
    output_mask = mask_d_out & mask_h_out & mask_w_out
    
    output_mask_reshaped = output_mask.reshape(BLOCK_D * BLOCK_H * BLOCK_W)
    
    tl.store(
        out_ptr + output_indices.reshape(BLOCK_D * BLOCK_H * BLOCK_W),
        acc.reshape(BLOCK_D * BLOCK_H * BLOCK_W),
        mask=output_mask_reshaped
    )


def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Triton-based 3D convolution implementation.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_depth, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections (must be 1 for this implementation)
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_depth, out_height, out_width)
    """
    assert groups == 1, "Group convolution not supported in this Triton kernel"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    N, C, D, H, W = x.shape
    OutC, _, Kd, Kh, Kw = weight.shape
    
    # Calculate output dimensions
    Od = (D + 2 * padding - dilation * (Kd - 1) - 1) // stride + 1
    Oh = (H + 2 * padding - dilation * (Kh - 1) - 1) // stride + 1
    Ow = (W + 2 * padding - dilation * (Kw - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((N, OutC, Od, Oh, Ow), dtype=x.dtype, device=x.device)
    
    # Define block sizes (tunable parameters for performance)
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4
    
    # Calculate grid dimensions
    grid = (
        N,  # batch size
        OutC,  # output channels
        (Od + BLOCK_D - 1) // BLOCK_D,  # output depth blocks
        (Oh + BLOCK_H - 1) // BLOCK_H,  # output height blocks
        (Ow + BLOCK_W - 1) // BLOCK_W   # output width blocks
    )
    
    # Launch the kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        N, C, D, H, W,
        OutC, Kd, Kh, Kw,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        Od, Oh, Ow,
        BLOCK_C=C,  # Not used in current implementation but kept for interface
        BLOCK_K=1,  # Not used in current implementation
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D convolution model using Triton kernels.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias
        
        # Create the weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and bias using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        # Ensure input is on CUDA and contiguous
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        
        # Run the Triton convolution
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )