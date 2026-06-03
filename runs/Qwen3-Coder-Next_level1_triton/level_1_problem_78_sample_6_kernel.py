import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def transposed_conv2d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H_in, W_in)
    w_ptr,  # Weight tensor: (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (B, C_out, H_out, W_out)
    B, C_in, C_out, K_h, K_w, 
    H_in, W_in, H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs for batch and output channels
    batch_id = tl.program_id(0)
    out_c_id = tl.program_id(1)
    
    # Calculate output position
    out_h_start = tl.program_id(2) * BLOCK_SIZE_H
    out_w_start = tl.program_id(3) * BLOCK_SIZE_W
    
    # Create ranges for output height and width
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for output dimensions
    out_h_mask = out_h_offsets < H_out
    out_w_mask = out_w_offsets < W_out
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for in_c in range(C_in):
        for kh in range(K_h):
            # Calculate input height position
            in_h = out_h_start * stride_h + kh - pad_h
            in_h_offsets = in_h + tl.arange(0, BLOCK_SIZE_H)
            in_h_mask = (in_h_offsets >= 0) & (in_h_offsets < H_in) & out_h_mask
            
            for kw in range(K_w):
                # Calculate input width position
                in_w = out_w_start * stride_w + kw - pad_w
                in_w_offsets = in_w + tl.arange(0, BLOCK_SIZE_W)
                in_w_mask = (in_w_offsets >= 0) & (in_w_offsets < W_in) & out_w_mask
                
                # Combined mask for valid positions
                valid_mask = in_h_mask[:, None] & in_w_mask[None, :]
                
                # Load input values if valid
                if in_h >= 0 and in_h < H_in and in_w >= 0 and in_w < W_in:
                    # Input index: [batch, in_c, in_h, in_w]
                    x_offset = (batch_id * C_in * H_in * W_in + 
                               in_c * H_in * W_in + 
                               in_h * W_in + in_w)
                    
                    # Load input with mask
                    x_val = tl.load(x_ptr + x_offset, mask=valid_mask, other=0.0)
                    
                    # Weight index: [in_c, out_c_id, kh, kw]
                    w_offset = (in_c * C_out * K_h * K_w + 
                               out_c_id * K_h * K_w + 
                               kh * K_w + kw)
                    
                    # Load weight
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_id)
        acc += bias
    
    # Store result
    out_offset = (batch_id * C_out * H_out * W_out + 
                 out_c_id * H_out * W_out + 
                 out_h_offsets[:, None] * W_out + out_w_offsets[None, :])
    
    out_mask = out_h_mask[:, None] & out_w_mask[None, :]
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_transposed_conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0)):
    """
    Triton-based transposed 2D convolution implementation.
    """
    # Extract dimensions
    B, C_in, H_in, W_in = x.shape
    _, C_out, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + K_h
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + K_w
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output tensor
    out = torch.empty(B, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Define block sizes for optimization
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_COUT = 8
    BLOCK_SIZE_CIN = 8
    BLOCK_SIZE_KH = 3
    BLOCK_SIZE_KW = 7
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    
    # Calculate grid dimensions
    grid = (
        B // BLOCK_SIZE_B,
        C_out // BLOCK_SIZE_COUT,
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
    # Launch kernel
    transposed_conv2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, K_h, K_w,
        H_in, W_in, H_out, W_out,
        stride[0], stride[1],
        padding[0], padding[1],
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
        BLOCK_SIZE_CIN=BLOCK_SIZE_CIN,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        
        # Create weight and bias parameters (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights (matching PyTorch's default initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D transposed convolution using Triton.
        """
        return triton_transposed_conv2d(
            x, self.weight, self.bias, 
            stride=self.stride, padding=self.padding
        )
    
    def _apply(self, fn):
        # Ensure parameters are moved to correct device
        super()._apply(fn)
        if self.bias is not None:
            self.bias = nn.Parameter(fn(self.bias.data))
        return self


import math