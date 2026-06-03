import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, H, W)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (N, C_out, H_out, W_out)
    n_elements,  # Total elements in output
    # Tensor dimensions
    N: tl.constexpr,  # Batch size
    C_in: tl.constexpr,  # Input channels
    H: tl.constexpr,  # Input height
    W: tl.constexpr,  # Input width
    C_out: tl.constexpr,  # Output channels
    K_h: tl.constexpr,  # Kernel height
    K_w: tl.constexpr,  # Kernel width
    stride_h: tl.constexpr,  # Stride height
    stride_w: tl.constexpr,  # Stride width
    pad_h: tl.constexpr,  # Padding height
    pad_w: tl.constexpr,  # Padding width
    dil_h: tl.constexpr,  # Dilation height
    dil_w: tl.constexpr,  # Dilation width
    # Output dimensions (calculated)
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    # Block sizes for tiling
    BLOCK_N: tl.constexpr,
    BLOCK_C_out: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Program IDs
    batch_idx = tl.program_id(0) // (H_out * W_out)
    hw_idx = tl.program_id(0) % (H_out * W_out)
    out_h = hw_idx // W_out
    out_w = hw_idx % W_out
    
    # Calculate input position corresponding to output position
    in_h = out_h * stride_h - pad_h
    in_w = out_w * stride_w - pad_w
    
    # Output block offset
    c_out_block_start = tl.program_id(1) * BLOCK_C_out
    c_out_offsets = c_out_block_start + tl.arange(0, BLOCK_C_out)
    c_out_mask = c_out_offsets < C_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C_out,), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for c_in in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate input position with dilation
                curr_h = in_h + kh * dil_h
                curr_w = in_w + kw * dil_w
                
                # Check bounds for input position
                in_bounds = (curr_h >= 0) & (curr_h < H) & (curr_w >= 0) & (curr_w < W)
                
                if tl.static_cast(tl.int1, in_bounds):
                    # Calculate input pointer offset
                    x_offset = batch_idx * C_in * H * W + c_in * H * W + curr_h * W + curr_w
                    # Load input value
                    x_val = tl.load(x_ptr + x_offset)
                    
                    # Calculate weight pointer offset
                    w_offset = c_out_block_start * C_in * K_h * K_w + c_in * K_h * K_w + kh * K_w + kw
                    # Load weight value
                    w_val = tl.load(w_ptr + w_offset, mask=c_out_mask)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_offset = c_out_block_start + tl.arange(0, BLOCK_C_out)
        bias_val = tl.load(b_ptr + bias_offset, mask=c_out_mask)
        acc += bias_val
    
    # Store result
    out_offset = batch_idx * C_out * H_out * W_out + c_out_offsets * H_out * W_out + out_h * W_out + out_w
    tl.store(out_ptr + out_offset, acc.to(tl.float32), mask=c_out_mask)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1
) -> torch.Tensor:
    """
    Triton implementation of 2D convolution.
    Note: This implementation assumes groups=1 for simplicity.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    # Calculate output dimensions
    H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes for tiling
    BLOCK_C_out = 32  # Tunable: channels per block
    # Calculate grid dimensions
    grid = (N * H_out * W_out, triton.cdiv(C_out, BLOCK_C_out))
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        N * C_out * H_out * W_out,
        N, C_in, H, W, C_out, K_h, K_w,
        stride_h, stride_w, pad_h, pad_w,
        dil_h, dil_w,
        H_out, W_out,
        BLOCK_N=1,
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_KH=1,
        BLOCK_KW=1,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
    
    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}'