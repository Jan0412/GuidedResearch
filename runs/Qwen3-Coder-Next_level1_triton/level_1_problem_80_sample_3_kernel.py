import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias tensor pointer (can be None)
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    in_h, in_w,  # Input height and width
    out_h, out_w,  # Output height and width
    k_h, k_w,  # Kernel height and width
    stride_h, stride_w,  # Stride
    pad_h, pad_w,  # Padding
    dil_h, dil_w,  # Dilation
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get program IDs for output spatial positions
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate output coordinates
    out_y = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_x = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid output coordinates
    mask_y = out_y < out_h
    mask_x = out_x < out_w
    mask = mask_y[:, None] & mask_x[None, :]
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Compute input coordinates corresponding to this output position
    in_y_start = out_y * stride_h - pad_h
    in_x_start = out_x * stride_w - pad_w
    
    # Loop over kernel height
    for kh in range(k_h):
        in_y = in_y_start + kh * dil_h
        mask_y_k = (in_y >= 0) & (in_y < in_h)
        
        # Loop over kernel width
        for kw in range(k_w):
            in_x = in_x_start + kw * dil_w
            mask_x_k = (in_x >= 0) & (in_x < in_w)
            mask_k = mask_y_k[:, None] & mask_x_k[None, :]
            
            # Load input values for this kernel position
            # Reshape to (BLOCK_SIZE_H * BLOCK_SIZE_W, in_channels)
            in_offsets_y = in_y[:, None] * in_w + in_x[None, :]
            in_offsets = (
                pid_b * (in_channels * in_h * in_w) +
                tl.arange(0, BLOCK_SIZE_H)[:, None] * (in_w * in_channels) +
                in_offsets_y[:, :, None] * in_channels +
                tl.arange(0, BLOCK_SIZE_C)[None, None, :]
            )
            
            # Create mask for input loading
            mask_in = (
                (tl.arange(0, BLOCK_SIZE_H)[:, None, None] < out_h) &
                (tl.arange(0, BLOCK_SIZE_W)[None, :, None] < out_w) &
                (tl.arange(0, BLOCK_SIZE_C)[None, None, :] < in_channels)
            )
            
            # Transpose for efficient loading: we want input[batch, c, y, x]
            # But our offsets are structured differently. Let's restructure:
            # in_offsets should be [block_h, block_w, c]
            # in_y is [block_h], in_x is [block_w]
            
            # For each output position, load input values
            # Simplified approach: iterate over channels and compute dot product
            
    # Alternative approach: compute convolution as matrix multiplication
    # For each output position (oh, ow), compute sum over c, kh, kw of x[b, c, oh*sh + kh*dh, ow*sw + kw*dw] * w[oc, c, kh, kw]
    
    # Let's restructure to a more efficient implementation
    # Process one output position per program, but use vectorized loads where possible
    
    # Get current batch index
    batch_offset = pid_b * (in_channels * in_h * in_w)
    
    # Loop over output channels
    for oc in range(out_channels):
        # Initialize accumulator for this output channel
        acc_oc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        
        # Loop over input channels in blocks
        for ic_block_start in range(0, in_channels, BLOCK_SIZE_C):
            ic_block_end = tl.minimum(ic_block_start + BLOCK_SIZE_C, in_channels)
            ic_block_size = ic_block_end - ic_block_start
            
            # Create range for input channels
            ic_offsets = ic_block_start + tl.arange(0, BLOCK_SIZE_C)
            ic_mask = ic_offsets < in_channels
            
            # Process this channel block
            for kh in range(k_h):
                in_y = out_y * stride_h + kh * dil_h - pad_h
                mask_y_valid = (in_y >= 0) & (in_y < in_h)
                
                for kw in range(k_w):
                    in_x = out_x * stride_w + kw * dil_w - pad_w
                    mask_x_valid = (in_x >= 0) & (in_x < in_w)
                    
                    # Compute input offset
                    # Input shape: (batch, in_channels, in_h, in_w)
                    input_offsets = (
                        batch_offset +
                        in_y[:, None] * in_w * in_channels +
                        in_x[None, :] * in_channels +
                        ic_offsets[None, None, :]
                    )
                    
                    # Compute weight offset
                    # Weight shape: (out_channels, in_channels, k_h, k_w)
                    weight_offsets = (
                        oc * (in_channels * k_h * k_w) +
                        ic_offsets[:, None, None] * (k_h * k_w) +
                        kh * k_w +
                        kw
                    )
                    
                    # Load input and weight values
                    input_values = tl.load(
                        x_ptr + input_offsets,
                        mask=mask_y_valid[:, None] & mask_x_valid[None, :] & ic_mask[None, None, :],
                        other=0.0
                    )
                    
                    weight_values = tl.load(
                        w_ptr + weight_offsets,
                        mask=ic_mask[:, None, None]
                    )
                    
                    # Accumulate the product
                    acc_oc += tl.sum(input_values * weight_values[None, :, :], axis=1)
        
        # Add bias if present
        if b_ptr is not None:
            bias = tl.load(b_ptr + oc)
            acc_oc += bias
        
        # Store result
        out_offset = (
            pid_b * (out_channels * out_h * out_w) +
            oc * (out_h * out_w) +
            out_y[:, None] * out_w +
            out_x[None, :]
        )
        
        tl.store(
            out_ptr + out_offset,
            acc_oc,
            mask=mask
        )


def triton_conv2d(x, weight, bias=None, stride=1, padding=(0, 0), dilation=(1, 1)):
    """
    Triton-based 2D convolution with support for dilation, padding, and asymmetric kernels.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, k_h, k_w = weight.shape
    
    # Calculate output dimensions
    stride_h = stride_w = stride if isinstance(stride, int) else stride
    pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
    dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    
    out_h = (in_h + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_C = 8
    
    grid = (
        triton.cdiv(out_h, BLOCK_SIZE_H),
        triton.cdiv(out_w, BLOCK_SIZE_W),
        batch_size,
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with Triton-based 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Register the convolution parameters and weights
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights using Kaiming initialization."""
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
            padding=self.padding,
            dilation=self.dilation
        )


# Import math for initialization
import math