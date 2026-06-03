import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to input/output tensors
    x_ptr,        # [B, IC, D, H, W]
    w_ptr,        # [IC, OC, kD, kH, kW]
    bias_ptr,     # [OC] (optional, can be None)
    out_ptr,      # [B, OC, D_out, H_out, W_out]
    # Tensor dimensions
    B, IC, D, H, W,
    OC, 
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    # Output dimensions
    out_D, out_H, out_W,
    # Block sizes for parallelization
    BLOCK_B: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_KD: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
    # Number of threads per dimension
    threads_per_batch: tl.constexpr,
    threads_per_channel: tl.constexpr,
):
    # Program IDs
    batch_id = tl.program_id(0)
    oc_id = tl.program_id(1)
    
    # Compute output position
    out_d = tl.program_id(2) // (out_H * out_W)
    rem = tl.program_id(2) % (out_H * out_W)
    out_h = rem // out_W
    out_w = rem % out_W
    
    # Compute the range of batch, output channel, and output position
    offsets_b = batch_id * threads_per_batch + tl.arange(0, BLOCK_B)
    offsets_oc = oc_id * threads_per_channel + tl.arange(0, BLOCK_OC)
    
    # Broadcast to match dimensions
    b_range = offsets_b[:, None, None, None, None] < B
    oc_range = offsets_oc[None, :, None, None, None] < OC
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_B, BLOCK_OC, 1, 1, 1), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for ic in range(IC):
        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    # Compute input position from output position
                    in_d = out_d + kd - pad_d
                    in_h = out_h + kh - pad_h
                    in_w = out_w + kw - pad_w
                    
                    # Check if input position is valid
                    valid_in = (in_d >= 0) & (in_d < D) & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
                    
                    if valid_in:
                        # Compute indices for input tensor
                        in_indices = tl.reshape(
                            batch_id * (IC * D * H * W) + 
                            ic * (D * H * W) + 
                            in_d * (H * W) + 
                            in_h * W + 
                            in_w,
                            (BLOCK_B, 1, 1, 1, 1)
                        )
                        
                        # Load input value
                        x_val = tl.load(
                            x_ptr + in_indices,
                            mask=b_range,
                            other=0.0
                        )
                        
                        # Compute weight index
                        w_indices = tl.reshape(
                            ic * (OC * kD * kH * kW) + 
                            oc_id * (kD * kH * kW) + 
                            kd * (kH * kW) + 
                            kh * kW + 
                            kw,
                            (1, BLOCK_OC, 1, 1, 1)
                        )
                        
                        # Load weight value
                        w_val = tl.load(
                            w_ptr + w_indices,
                            mask=oc_range,
                            other=0.0
                        )
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets_oc, mask=offsets_oc < OC, other=0.0)
        acc += bias[None, :, None, None, None]
    
    # Compute output index
    out_indices = batch_id * (OC * out_D * out_H * out_W) + \
                  offsets_oc[None, :, None, None, None] * (out_D * out_H * out_W) + \
                  out_d * (out_H * out_W) + \
                  out_h * out_W + \
                  out_w
    
    # Store result
    acc = acc.to(tl.float32)
    tl.store(
        out_ptr + out_indices,
        acc,
        mask=b_range & oc_range
    )


def triton_conv_transpose3d(x, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1):
    """
    Triton implementation of ConvTranspose3d
    
    Args:
        x: Input tensor [B, IC, D, H, W]
        weight: Weight tensor [IC, OC, kD, kH, kW]
        bias: Optional bias tensor [OC]
        stride: Stride (d, h, w)
        padding: Padding (d, h, w)
        output_padding: Output padding (d, h, w)
        groups: Number of groups (must be 1 for this implementation)
    """
    assert groups == 1, "Groups > 1 not supported in this Triton kernel"
    
    # Extract dimensions
    B, IC, D, H, W = x.shape
    IC2, OC, kD, kH, kW = weight.shape
    assert IC == IC2, f"Input channels mismatch: {IC} vs {IC2}"
    
    # Compute output dimensions
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding
    
    out_D = (D - 1) * stride_d - 2 * pad_d + kD + out_pad_d
    out_H = (H - 1) * stride_h - 2 * pad_h + kH + out_pad_h
    out_W = (W - 1) * stride_w - 2 * pad_w + kW + out_pad_w
    
    # Create output tensor
    out = torch.empty(B, OC, out_D, out_H, out_W, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    # For simplicity, we'll use a 3D grid: [batch, output_channels, output_positions]
    # Output positions = out_D * out_H * out_W
    grid = (B, OC, out_D * out_H * out_W)
    
    # Block sizes - tune these for your GPU
    BLOCK_B = 1
    BLOCK_OC = 16
    BLOCK_IC = 1
    BLOCK_KD = 1
    BLOCK_KH = 1
    BLOCK_KW = 1
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, IC, D, H, W,
        OC,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        out_D, out_H, out_W,
        BLOCK_B=BLOCK_B,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        BLOCK_KD=BLOCK_KD,
        BLOCK_KH=BLOCK_KH,
        BLOCK_KW=BLOCK_KW,
        threads_per_batch=1,
        threads_per_channel=1
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), 
                 padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Create weight and bias parameters (same as nn.ConvTranspose3d)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Reset parameters similar to nn.ConvTranspose3d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Call Triton kernel
        return triton_conv_transpose3d(
            x, weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )
    
    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, output_padding={self.output_padding}, '
                f'groups={self.groups}, bias={self.bias is not None}')