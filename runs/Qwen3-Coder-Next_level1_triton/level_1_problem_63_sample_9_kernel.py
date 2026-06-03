import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,              # Input tensor (N, C, H, W)
    w_ptr,              # Weight tensor (OC, IC, KH, KW)
    b_ptr,              # Bias tensor (OC,) - can be None
    out_ptr,            # Output tensor (N, OC, OH, OW)
    N, C, H, W,         # Input dimensions
    OC, KH, KW,         # Output channels and kernel dimensions
    stride, padding, dilation,
    OH, OW,             # Output dimensions
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)
    pid_oc = tl.program_id(3)
    
    # Calculate output position
    out_h = pid_oh * BLOCK_H
    out_w = pid_ow * BLOCK_W
    out_c = pid_oc * BLOCK_OC
    
    # Create output offset arrays
    oh_offsets = tl.arange(0, BLOCK_H)
    ow_offsets = tl.arange(0, BLOCK_W)
    oc_offsets = tl.arange(0, BLOCK_OC)
    
    oh_mask = (out_h + oh_offsets) < OH
    ow_mask = (out_w + ow_offsets) < OW
    oc_mask = (out_c + oc_offsets) < OC
    
    # Initialize output accumulator
    output = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_OC), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for ic in range(C):
        for kh in range(KH):
            # Calculate input h position with dilation
            in_h = out_h * stride + kh * dilation - padding
            h_mask = (in_h >= 0) & (in_h < H)
            
            for kw in range(KW):
                # Calculate input w position with dilation
                in_w = out_w * stride + kw * dilation - padding
                w_mask = (in_w >= 0) & (in_w < W)
                
                # Load input values if within bounds
                x_offset = ((pid_n * C * H * W) + 
                           (ic * H * W) + 
                           (in_h * W) + 
                           in_w)
                
                # Load weight values
                w_offset = ((out_c * C * KH * KW) + 
                           (ic * KH * KW) + 
                           (kh * KW) + 
                           kw)
                
                # Use masks for boundary conditions
                x_vals = tl.load(
                    x_ptr + x_offset,
                    mask=(h_mask[:, None] & w_mask[None, :]),
                    other=0.0
                )
                
                w_vals = tl.load(
                    w_ptr + w_offset,
                    mask=oc_mask[:, None, None],
                    other=0.0
                )
                
                # Multiply and accumulate
                output += x_vals[None, :, :] * w_vals[:, :, :]
    
    # Add bias if available
    if b_ptr is not None:
        b_offset = out_c + tl.arange(0, BLOCK_OC)
        b_vals = tl.load(b_ptr + b_offset, mask=oc_mask, other=0.0)
        output += b_vals[None, None, :]
    
    # Store results
    out_offset = ((pid_n * OC * OH * OW) + 
                 ((out_c + tl.arange(0, BLOCK_OC)[:, None, None]) * OH * OW) + 
                 ((out_h + oh_offsets[None, :, None]) * OW) + 
                 (out_w + ow_offsets[None, None, :]))
    
    tl.store(
        out_ptr + out_offset,
        output,
        mask=(oc_mask[:, None, None] & oh_mask[None, :, None] & ow_mask[None, None, :])
    )


def triton_conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Custom Triton implementation of 2D convolution.
    
    Args:
        x: Input tensor (N, C, H, W)
        weight: Weight tensor (OC, IC, KH, KW)
        bias: Bias tensor (OC,) or None
        stride, padding, dilation, groups: Convolution parameters
    
    Returns:
        Output tensor
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    N, C, H, W = x.shape
    OC, IC, KH, KW = weight.shape
    
    # Calculate output dimensions
    OH = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)
    
    # Define block sizes for tiling (tunable parameters for performance)
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_OC = 32
    BLOCK_KH = 1
    BLOCK_KW = 1
    BLOCK_IC = 1
    
    # Grid dimensions
    grid = (
        N,                    # batch dimension
        (OH + BLOCK_H - 1) // BLOCK_H,  # output height
        (OW + BLOCK_W - 1) // BLOCK_W,  # output width  
        (OC + BLOCK_OC - 1) // BLOCK_OC, # output channels
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        N, C, H, W,
        OC, KH, KW,
        stride, padding, dilation,
        OH, OW,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_KH=BLOCK_KH,
        BLOCK_KW=BLOCK_KW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same way as the original model
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), 
                               stride=stride, padding=padding, dilation=dilation, 
                               groups=groups, bias=bias)
        
        # Store parameters for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use our custom Triton implementation
        return triton_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias if self.has_bias else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )