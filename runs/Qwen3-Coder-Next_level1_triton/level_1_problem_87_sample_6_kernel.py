import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def pointwise_conv1d_kernel(
    x_ptr,  # Input tensor pointer: (B, C_in, H, W)
    w_ptr,  # Weight tensor pointer: (C_out, C_in, 1, 1)
    b_ptr,  # Bias tensor pointer: (C_out,) or None
    out_ptr,  # Output tensor pointer: (B, C_out, H, W)
    B, C_in, H, W, C_out,
    stride_x, stride_x_c, stride_x_h, stride_x_w,
    stride_w, stride_w_c, stride_w_kh, stride_w_kw,
    stride_out, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Compute batch index
    batch_idx = pid_b
    # Compute output channel index
    c_out_idx = pid_c_out * BLOCK_SIZE_COUT + tl.arange(0, BLOCK_SIZE_COUT)
    c_out_mask = c_out_idx < C_out
    
    # Compute spatial index (flattened H*W)
    hw_idx = pid_hw * BLOCK_SIZE_HW
    hw_range = tl.arange(0, BLOCK_SIZE_HW)
    hw_offsets = hw_idx + hw_range
    h_offsets = hw_offsets // W
    w_offsets = hw_offsets % W
    hw_mask = hw_offsets < H * W
    
    # Create masks for output
    out_mask = c_out_mask[:, None] & hw_mask[None, :]
    
    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_COUT, BLOCK_SIZE_HW), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for off_c_in in range(0, C_in, BLOCK_SIZE_CIN):
        c_in_range = off_c_in + tl.arange(0, BLOCK_SIZE_CIN)
        c_in_mask = c_in_range < C_in
        
        # Create mask for input channel and spatial positions
        x_mask = c_in_mask[:, None] & hw_mask[None, :]
        
        # Load input block: (BLOCK_SIZE_CIN, BLOCK_SIZE_HW)
        x_block = tl.load(
            x_ptr + 
            batch_idx * stride_x + 
            c_in_range[:, None] * stride_x_c + 
            h_offsets[None, :] * stride_x_h + 
            w_offsets[None, :] * stride_x_w,
            mask=x_mask,
            other=0.0
        )
        
        # Load weight block: (BLOCK_SIZE_COUT, BLOCK_SIZE_CIN)
        w_block = tl.load(
            w_ptr + 
            c_out_idx[:, None] * stride_w + 
            c_in_range[None, :] * stride_w_c,
            mask=c_out_mask[:, None] & c_in_mask[None, :],
            other=0.0
        )
        
        # Compute partial dot product
        # acc += w_block @ x_block^T
        acc += tl.dot(w_block, x_block, allow_tf32=False)
    
    # Convert accumulator to output dtype if needed
    acc = acc.to(x_ptr.dtype.element_ty)
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_idx, mask=c_out_mask, other=0.0)
        acc += bias[:, None]
    
    # Store output
    tl.store(
        out_ptr +
        batch_idx * stride_out +
        c_out_idx[:, None] * stride_out_c +
        h_offsets[None, :] * stride_out_h +
        w_offsets[None, :] * stride_out_w,
        acc,
        mask=out_mask
    )


def triton_pointwise_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Performs pointwise 2D convolution using Triton kernel.
    Pointwise convolution is a 1x1 convolution that operates across channels.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, 1, 1)
        bias: Optional bias tensor of shape (out_channels,)
    
    Returns:
        Output tensor of shape (batch_size, out_channels, height, width)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Output tensor
    out = torch.empty((B, C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Compute strides
    stride_x = x.stride(0)
    stride_x_c = x.stride(1)
    stride_x_h = x.stride(2)
    stride_x_w = x.stride(3)
    
    stride_w = weight.stride(0)
    stride_w_c = weight.stride(1)
    stride_w_kh = weight.stride(2)
    stride_w_kw = weight.stride(3)
    
    stride_out = out.stride(0)
    stride_out_c = out.stride(1)
    stride_out_h = out.stride(2)
    stride_out_w = out.stride(3)
    
    # Grid dimensions: (batch_size, num_c_out_blocks, num_hw_blocks)
    BLOCK_SIZE_CIN = 32
    BLOCK_SIZE_COUT = 32
    BLOCK_SIZE_HW = 256
    
    grid = (
        B,
        triton.cdiv(C_out, BLOCK_SIZE_COUT),
        triton.cdiv(H * W, BLOCK_SIZE_HW)
    )
    
    # Launch kernel
    pointwise_conv1d_kernel[grid](
        x, weight, bias, out,
        B, C_in, H, W, C_out,
        stride_x, stride_x_c, stride_x_h, stride_x_w,
        stride_w, stride_w_c, stride_w_kh, stride_w_kw,
        stride_out, stride_out_c, stride_out_h, stride_out_w,
        BLOCK_SIZE_CIN=BLOCK_SIZE_CIN,
        BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized pointwise 2D convolution using Triton kernel.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias similar to nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Kaiming initialization similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return triton_pointwise_conv1d(x, self.weight, self.bias)