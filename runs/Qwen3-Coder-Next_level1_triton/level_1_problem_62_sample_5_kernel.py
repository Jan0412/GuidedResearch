import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Input tensor pointer (N, C, H, W)
    weight_ptr,  # Weight tensor pointer (out_channels, in_channels, kH, kW)
    bias_ptr,  # Bias pointer (out_channels) - can be None
    output_ptr,  # Output tensor pointer (N, out_channels, H_out, W_out)
    N,  # Batch size
    C,  # Input channels
    H, W,  # Input height and width
    out_H, out_W,  # Output height and width
    kH, kW,  # Kernel height and width
    stride,  # Stride
    padding,  # Padding
    dilation,  # Dilation
    out_channels,  # Number of output channels
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
    BLOCK_SIZE_OUT_C: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_c_block = tl.program_id(1)
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)
    
    # Calculate output channel range for this block
    out_c_start = out_c_block * BLOCK_SIZE_OUT_C
    out_c_end = tl.minimum(out_c_start + BLOCK_SIZE_OUT_C, out_channels)
    
    # Create output channel mask
    out_c_mask = (out_c_start + tl.arange(0, BLOCK_SIZE_OUT_C)) < out_channels
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_OUT_C,), tl.float32)
    
    # Loop over input channels in blocks
    for c_idx in range(0, C, BLOCK_SIZE_C):
        c_end = tl.minimum(c_idx + BLOCK_SIZE_C, C)
        c_block = c_idx + tl.arange(0, BLOCK_SIZE_C)
        c_mask = c_block < C
        
        # Loop over kernel height
        for kh_idx in range(0, kH, BLOCK_SIZE_KH):
            kh_end = tl.minimum(kh_idx + BLOCK_SIZE_KH, kH)
            kh_block = kh_idx + tl.arange(0, BLOCK_SIZE_KH)
            kh_mask = kh_block < kH
            
            # Loop over kernel width
            for kw_idx in range(0, kW, BLOCK_SIZE_KW):
                kw_end = tl.minimum(kw_idx + BLOCK_SIZE_KW, kW)
                kw_block = kw_idx + tl.arange(0, BLOCK_SIZE_KW)
                kw_mask = kw_block < kW
                
                # Calculate input coordinates
                in_h = out_h * stride + kh_idx * dilation - padding
                in_w = out_w * stride + kw_idx * dilation - padding
                
                # Create masks for valid input coordinates
                h_valid = (in_h >= 0) & (in_h < H)
                w_valid = (in_w >= 0) & (in_w < W)
                
                # Load input values
                input_offsets = (
                    batch_idx * (C * H * W) +
                    c_block[:, None, None] * (H * W) +
                    (in_h + kh_block[None, :, None]) * W +
                    (in_w + kw_block[None, None, :])
                )
                
                # Reshape masks for proper broadcasting
                c_mask_reshaped = c_mask[:, None, None]
                kh_mask_reshaped = kh_mask[None, :, None]
                kw_mask_reshaped = kw_mask[None, None, :]
                valid_mask = h_valid & w_valid & c_mask_reshaped & kh_mask_reshaped & kw_mask_reshaped
                
                input_vals = tl.load(input_ptr + input_offsets, mask=valid_mask, other=0.0)
                
                # Load weight values
                weight_offsets = (
                    (out_c_start + tl.arange(0, BLOCK_SIZE_OUT_C)[:, None, None, None]) * (C * kH * kW) +
                    c_block[None, :, None, None] * (kH * kW) +
                    kh_block[None, None, :, None] * kW +
                    kw_block[None, None, None, :]
                )
                weight_vals = tl.load(weight_ptr + weight_offsets, mask=out_c_mask[:, None, None, None] & c_mask_reshaped & kh_mask_reshaped & kw_mask_reshaped, other=0.0)
                
                # Accumulate convolution result
                acc += tl.sum(input_vals * weight_vals, axis=(1, 2, 3))
    
    # Add bias if present
    if bias_ptr is not None:
        bias_offsets = out_c_start + tl.arange(0, BLOCK_SIZE_OUT_C)
        bias_mask = bias_offsets < out_channels
        bias_vals = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
        acc += bias_vals
    
    # Store output
    output_offsets = (
        batch_idx * (out_channels * out_H * out_W) +
        (out_c_start + tl.arange(0, BLOCK_SIZE_OUT_C)) * (out_H * out_W) +
        out_h * out_W +
        out_w
    )
    tl.store(output_ptr + output_offsets, acc.to(tl.float32), mask=out_c_mask)


def triton_conv2d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 2D convolution.
    
    Note: This implementation assumes groups=1 for simplicity.
    For groups > 1, the implementation would need to be extended.
    """
    assert groups == 1, "Triton conv2d currently only supports groups=1"
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    N, C, H, W = input_tensor.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    out_H = (H + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    out_W = (W + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(N, out_channels, out_H, out_W, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Define block sizes (tunable parameters for performance)
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_OUT_C = 8
    BLOCK_SIZE_C = 8
    BLOCK_SIZE_KH = 3
    BLOCK_SIZE_KW = 3
    
    # Calculate grid dimensions
    grid = (N, (out_channels + BLOCK_SIZE_OUT_C - 1) // BLOCK_SIZE_OUT_C, out_H, out_W)
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor, weight, bias, output,
        N, C, H, W, out_H, out_W,
        kH, kW, stride, padding, dilation, out_channels,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
        BLOCK_SIZE_OUT_C=BLOCK_SIZE_OUT_C,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton convolution kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        
        # Create parameter tensors for weights and bias
        kH, kW = self.kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kH, kW))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding[0],  # Using only the first padding value for simplicity
            dilation=self.dilation[0],  # Using only the first dilation value for simplicity
            groups=self.groups
        )