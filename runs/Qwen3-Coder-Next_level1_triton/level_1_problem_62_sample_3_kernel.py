import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, H, W)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,)
    out_ptr,  # Output tensor pointer (N, C_out, H_out, W_out)
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    n_channels_in: tl.constexpr,
    n_channels_out: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    input_h: tl.constexpr,
    input_w: tl.constexpr,
    output_h: tl.constexpr,
    output_w: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Batch index
    batch_idx = tl.program_id(0)
    # Output channel block index
    c_out_block_idx = tl.program_id(1)
    
    # Calculate output position
    c_out_start = c_out_block_idx * BLOCK_SIZE_C_OUT
    c_out = tl.arange(0, BLOCK_SIZE_C_OUT) + c_out_start
    c_out_mask = c_out < n_channels_out
    
    # Output spatial indices
    h_out = tl.program_id(2) // output_w
    w_out = tl.program_id(2) % output_w
    
    # Compute the starting position in the input for this output position
    h_in = h_out * stride - padding + tl.arange(0, kernel_h) * dilation
    w_in = w_out * stride - padding + tl.arange(0, kernel_w) * dilation
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for c_in_block in range(0, n_channels_in, BLOCK_SIZE_C_IN):
        c_in = tl.arange(0, BLOCK_SIZE_C_IN) + c_in_block
        c_in_mask = c_in < n_channels_in
        
        # Create masks for valid input positions
        h_in_valid = (h_in >= 0) & (h_in < input_h)
        w_in_valid = (w_in >= 0) & (w_in < input_w)
        valid_mask = h_in_valid[:, None] & w_in_valid[None, :]
        
        # Load input values
        # Input shape: (batch, channels, height, width)
        input_offsets = batch_idx * (n_channels_in * input_h * input_w) + \
                       c_in[:, None, None] * (input_h * input_w) + \
                       h_in[None, :, None] * input_w + \
                       w_in[None, None, :]
        input_offsets = tl.reshape(input_offsets, [BLOCK_SIZE_C_IN * kernel_h * kernel_w])
        
        # Flatten valid mask
        valid_flat = tl.reshape(valid_mask, [BLOCK_SIZE_C_IN * kernel_h * kernel_w])
        
        # Load input with padding handling
        x_block = tl.load(x_ptr + input_offsets, mask=valid_flat, other=0.0)
        x_block = tl.reshape(x_block, [BLOCK_SIZE_C_IN, kernel_h, kernel_w])
        
        # Load weights
        weight_offsets = c_out[:, None, None, None] * (n_channels_in * kernel_h * kernel_w) + \
                        c_in[None, :, None, None] * (kernel_h * kernel_w) + \
                        tl.arange(0, kernel_h)[None, None, :, None] * kernel_w + \
                        tl.arange(0, kernel_w)[None, None, None, :]
        weight_offsets = tl.reshape(weight_offsets, [BLOCK_SIZE_C_OUT * BLOCK_SIZE_C_IN * kernel_h * kernel_w])
        
        c_out_full = tl.arange(0, BLOCK_SIZE_C_OUT)[:, None, None, None] + c_out_start
        weight_mask = (c_out_full < n_channels_out) & (c_in[None, :, None, None] < n_channels_in)
        weight_mask = tl.reshape(weight_mask, [BLOCK_SIZE_C_OUT * BLOCK_SIZE_C_IN * kernel_h * kernel_w])
        
        w_block = tl.load(w_ptr + weight_offsets, mask=weight_mask, other=0.0)
        w_block = tl.reshape(w_block, [BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN, kernel_h, kernel_w])
        
        # Compute convolution contribution
        x_expanded = x_block[None, :, :, :]  # [1, C_in, K_h, K_w]
        w_expanded = w_block  # [C_out, C_in, K_h, K_w]
        
        # Multiply and accumulate
        prod = x_expanded * w_expanded
        acc += tl.sum(prod, axis=[1, 2, 3])
    
    # Apply bias if available
    if b_ptr is not None:
        bias_offsets = c_out
        bias = tl.load(b_ptr + bias_offsets, mask=c_out_mask, other=0.0)
        acc += bias
    
    # Store output
    out_offsets = batch_idx * (n_channels_out * output_h * output_w) + \
                 c_out * (output_h * output_w) + \
                 h_out * output_w + \
                 w_out
    tl.store(out_ptr + out_offsets, acc.to(tl.float32), mask=c_out_mask)


def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    """
    Triton implementation of 2D convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, dilation, groups: Convolution parameters
    
    Returns:
        Output tensor of shape (batch_size, out_channels, output_height, output_width)
    """
    batch_size, in_channels, input_h, input_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_h = (input_h + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    output_w = (input_w + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output tensor
    out = torch.empty(batch_size, out_channels, output_h, output_w, dtype=x.dtype, device=x.device)
    
    # Define block sizes for tiling
    BLOCK_SIZE_N = 1  # Batch size is processed per kernel
    BLOCK_SIZE_C_OUT = 32  # Output channel block size
    BLOCK_SIZE_C_IN = min(32, in_channels)  # Input channel block size
    BLOCK_SIZE_KH = kernel_h  # Kernel height
    BLOCK_SIZE_KW = kernel_w  # Kernel width
    
    # Grid configuration
    grid = (batch_size, 
            triton.cdiv(out_channels, BLOCK_SIZE_C_OUT), 
            output_h * output_w)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        stride=stride,
        padding=padding,
        dilation=dilation,
        n_channels_in=in_channels,
        n_channels_out=out_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        input_h=input_h,
        input_w=input_w,
        output_h=output_h,
        output_w=output_w,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias similar to nn.Conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights using kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution using Triton kernel.
        """
        return triton_conv2d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, 
                            self.groups)