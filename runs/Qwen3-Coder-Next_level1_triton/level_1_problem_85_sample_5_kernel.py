import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    w_ptr,  # Weight tensor pointer (C, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C,) or None
    out_ptr,  # Output tensor pointer (N, C, H_out, W_out)
    N, C, H, W,  # Input dimensions
    H_out, W_out,  # Output dimensions
    K_h: tl.constexpr, K_w: tl.constexpr,  # Kernel dimensions
    stride_h: tl.constexpr, stride_w: tl.constexpr,  # Stride
    padding_h: tl.constexpr, padding_w: tl.constexpr,  # Padding
    dilation_h: tl.constexpr, dilation_w: tl.constexpr,  # Dilation
    C_numel: tl.constexpr,  # Number of channels
    BLOCK_SIZE_H: tl.constexpr = 16,
    BLOCK_SIZE_W: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 1,
):
    # Get the program IDs
    batch_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    block_h = tl.program_id(2)
    block_w = tl.program_id(3)
    
    # Calculate output position
    out_h = block_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = block_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid output positions
    h_mask = out_h < H_out
    w_mask = out_w < W_out
    hw_mask = h_mask[:, None] & w_mask[None, :]
    
    # Calculate input position corresponding to this output
    in_h_start = out_h * stride_h - padding_h
    in_w_start = out_w * stride_w - padding_w
    
    # Compute one output element per (h, w) position
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kh in range(K_h):
        in_h = in_h_start + kh * dilation_h
        h_valid = (in_h >= 0) & (in_h < H)
        
        for kw in range(K_w):
            in_w = in_w_start + kw * dilation_w
            w_valid = (in_w >= 0) & (in_w < W)
            
            # Calculate input indices
            in_h_idx = in_h[:, None] if K_h > 1 else in_h[None, :]
            in_w_idx = in_w[None, :] if K_w > 1 else in_w[:, None]
            
            # Get kernel weight
            w_offset = c_idx * K_h * K_w + kh * K_w + kw
            weight = tl.load(w_ptr + w_offset)
            
            # Get input value
            input_offset = batch_idx * C * H * W + c_idx * H * W + in_h_idx * W + in_w_idx
            input_val = tl.load(x_ptr + input_offset, mask=(h_valid & w_valid)[:, :], other=0.0)
            
            # Accumulate
            acc += input_val * weight
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_idx)
        acc += bias
    
    # Store result
    out_offset = batch_idx * C * H_out * W_out + c_idx * H_out * W_out + out_h[:, None] * W_out + out_w[None, :]
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=hw_mask)


def triton_depthwise_conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Performs depthwise 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        weight: Weight tensor of shape (C, K_h, K_w)
        bias: Optional bias tensor of shape (C,)
        stride: Stride (stride_h, stride_w)
        padding: Padding (padding_h, padding_w)
        dilation: Dilation (dilation_h, dilation_w)
    
    Returns:
        Output tensor of shape (N, C, H_out, W_out)
    """
    N, C, H, W = x.shape
    C_w, K_h, K_w = weight.shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Calculate output dimensions
    H_out = (H + 2 * padding_h - dilation_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * padding_w - dilation_w * (K_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Define block sizes
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    # Calculate grid dimensions
    grid = (
        N,  # batch size
        C,  # number of channels
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,  # height blocks
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W   # width blocks
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        N, C, H, W,
        H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel using Triton.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        
        # Initialize weights and bias
        # For depthwise convolution, weight shape is (in_channels, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size_h, kernel_size_w))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights with kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = kernel_size_h * kernel_size_w * in_channels
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_depthwise_conv2d(
            x, self.weight, self.bias,
            stride=(self.stride_h, self.stride_w),
            padding=(self.padding_h, self.padding_w),
            dilation=(self.dilation_h, self.dilation_w)
        )


# Import math for initialization
import math