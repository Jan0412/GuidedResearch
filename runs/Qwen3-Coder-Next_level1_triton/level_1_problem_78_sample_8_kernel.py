import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv2d_kernel(
    x_ptr,              # Input tensor pointer: (B, C_in, H_in, W_in)
    w_ptr,              # Weight tensor pointer: (C_in, C_out, K_h, K_w)
    b_ptr,              # Bias tensor pointer: (C_out,) or None
    y_ptr,              # Output tensor pointer: (B, C_out, H_out, W_out)
    B: tl.constexpr,    # Batch size
    C_in: tl.constexpr, # Input channels
    C_out: tl.constexpr, # Output channels
    H_in: tl.constexpr, # Input height
    W_in: tl.constexpr, # Input width
    K_h: tl.constexpr,  # Kernel height
    K_w: tl.constexpr,  # Kernel width
    H_out: tl.constexpr, # Output height
    W_out: tl.constexpr, # Output width
    stride_h: tl.constexpr, # Stride height
    stride_w: tl.constexpr, # Stride width
    pad_h: tl.constexpr,    # Padding height
    pad_w: tl.constexpr,    # Padding width
    HAS_BIAS: tl.constexpr, # Whether bias is present
    BLOCK_SIZE: tl.constexpr = 128
):
    # Each program handles one output element: (batch, out_channel, out_h, out_w)
    # We'll use a 1D grid and compute the 4D indices
    idx = tl.program_id(0)
    
    # Compute 4D indices from linear index
    # Order: batch, out_channel, out_h, out_w
    tmp = idx
    out_w = tmp % W_out
    tmp //= W_out
    out_h = tmp % H_out
    tmp //= H_out
    out_c = tmp % C_out
    batch = tmp // C_out
    
    # Compute the starting position in input for this output position
    # For transposed convolution: in_h_start = out_h * stride_h - pad_h
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Accumulator for the result
    acc = 0.0
    
    # Iterate over input channels and kernel positions
    for c_in in range(C_in):
        for kh in range(K_h):
            in_h = in_h_start + kh
            # Check if input height is within bounds
            if 0 <= in_h < H_in:
                for kw in range(K_w):
                    in_w = in_w_start + kw
                    # Check if input width is within bounds
                    if 0 <= in_w < W_in:
                        # Compute indices for input, weight, and output
                        x_idx = batch * (C_in * H_in * W_in) + \
                                c_in * (H_in * W_in) + \
                                in_h * W_in + in_w
                        w_idx = c_in * (C_out * K_h * K_w) + \
                                out_c * (K_h * K_w) + \
                                kh * K_w + kw
                        acc += tl.load(x_ptr + x_idx) * tl.load(w_ptr + w_idx)
    
    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(b_ptr + out_c)
        acc += bias
    
    # Store the result
    y_idx = idx
    tl.store(y_ptr + y_idx, acc)


def triton_transposed_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                            stride: tuple = (1, 1), padding: tuple = (0, 0)) -> torch.Tensor:
    """
    Triton implementation of 2D transposed convolution.
    
    Args:
        x: Input tensor of shape (B, C_in, H_in, W_in)
        weight: Weight tensor of shape (C_in, C_out, K_h, K_w)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride (stride_h, stride_w)
        padding: Padding (pad_h, pad_w)
    
    Returns:
        Output tensor of shape (B, C_out, H_out, W_out)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    B, C_in, H_in, W_in = x.shape
    _, C_out, K_h, K_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h
    W_out = (W_in - 1) * stride_w - 2 * pad_w + K_w
    
    # Create output tensor
    y = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Total number of output elements
    total_elements = B * C_out * H_out * W_out
    BLOCK_SIZE = 128
    
    # Create grid
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    transposed_conv2d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out, H_in, W_in, K_h, K_w, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w,
        HAS_BIAS=bias is not None,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the weight and bias parameters directly
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Weight shape for ConvTranspose2d is (in_channels, out_channels, *kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size[0], kernel_size[1]))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters (using same initialization as PyTorch's ConvTranspose2d)
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming uniform initialization similar to PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_transposed_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding
        )


# Import math for initialization
import math