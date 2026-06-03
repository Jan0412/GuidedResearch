import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,           # Input tensor pointer (B, C_in, H, W)
    w_ptr,           # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,           # Bias tensor pointer (C_out,) or None
    y_ptr,           # Output tensor pointer (B, C_out, H_out, W_out)
    B, C_in, C_out,  # Batch size, input channels, output channels
    H, W,            # Input height, width
    K_h, K_w,        # Kernel height, width
    stride_h, stride_w,  # Stride
    pad_h, pad_w,        # Padding
    dil_h, dil_w,        # Dilation
    H_out, W_out,        # Output height, width
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_H: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_W: tl.constexpr,  # Block size for output width
):
    # Get program IDs
    pid_c = tl.program_id(0)  # Output channel block
    pid_b = tl.program_id(1)  # Batch block
    pid_h = tl.program_id(2)  # Output height block
    pid_w = tl.program_id(3)  # Output width block
    
    # Compute output feature coordinates
    out_h_start = pid_h * BLOCK_SIZE_H
    out_w_start = pid_w * BLOCK_SIZE_W
    
    # Compute input feature coordinates with padding and dilation
    # For each output position, compute the corresponding input position
    # We'll process one output pixel at a time for simplicity and correctness
    
    # Create output offsets for this block
    out_offsets_h = tl.arange(0, BLOCK_SIZE_H)
    out_offsets_w = tl.arange(0, BLOCK_SIZE_W)
    out_h = out_h_start + out_offsets_h[:, None]
    out_w = out_w_start + out_offsets_w[None, :]
    
    # Check bounds for output
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask = mask_h & mask_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Convolution: iterate over input channels and kernel positions
    for c_in in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                # Compute input position: out_pos = stride * in_pos + dilation * (kernel_pos - padding)
                in_h = out_h * stride_h + kh * dil_h - pad_h
                in_w = out_w * stride_w + kw * dil_w - pad_w
                
                # Check if input position is within bounds
                in_h_mask = (in_h >= 0) & (in_h < H)
                in_w_mask = (in_w >= 0) & (in_w < W)
                input_mask = in_h_mask & in_w_mask
                
                # Compute input pointer offset
                # Input layout: (B, C_in, H, W)
                in_offset = pid_b * (C_in * H * W) + c_in * (H * W) + in_h * W + in_w
                
                # Load input value with mask
                x_val = tl.load(x_ptr + in_offset, mask=input_mask & mask, other=0.0)
                
                # Compute weight pointer offset
                # Weight layout: (C_out, C_in, K_h, K_w)
                weight_offset = pid_c * (C_in * K_h * K_w) + c_in * (K_h * K_w) + kh * K_w + kw
                
                # Load weight value
                w_val = tl.load(w_ptr + weight_offset)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_offset = pid_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_H)[:, None] * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)[None, :]
        bias_offset = pid_c  # Bias is just [C_out]
        # Actually bias is 1D, so we need to broadcast
        bias_val = tl.load(b_ptr + pid_c)
        acc += bias_val
    
    # Store result
    y_offset = pid_b * (C_out * H_out * W_out) + pid_c * (H_out * W_out) + out_h * W_out + out_w
    tl.store(y_ptr + y_offset, acc.to(x_ptr.dtype.element_ty), mask=mask)

def triton_conv2d(x, weight, bias=None, stride=1, padding=(0, 0), dilation=(1, 1)):
    """
    Triton-based 2D convolution.
    
    Args:
        x: Input tensor of shape (B, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in, K_h, K_w)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride for convolution
        padding: Padding tuple (pad_h, pad_w)
        dilation: Dilation tuple (dil_h, dil_w)
    
    Returns:
        Output tensor of shape (B, C_out, H_out, W_out)
    """
    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    
    # Calculate output dimensions
    H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride + 1
    W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride + 1
    
    # Allocate output tensor
    y = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_M = 16  # Output channels per block
    BLOCK_SIZE_N = 1   # Batch size per block (1 for simplicity)
    BLOCK_SIZE_K = 8   # Input channels per block
    BLOCK_SIZE_H = 8   # Output height per block
    BLOCK_SIZE_W = 8   # Output width per block
    
    grid = (
        (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (B + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out,
        H, W,
        K_h, K_w,
        stride, stride,
        pad_h, pad_w,
        dil_h, dil_w,
        H_out, W_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return y

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.
    Uses optimized Triton kernel instead of PyTorch's native implementation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width). 
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (tuple, optional): Padding applied to the input (top/bottom, left/right). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        # Store parameters for Triton kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Ensure inputs are contiguous
        x = x.contiguous()
        
        # Get parameters from the original conv2d layer
        weight = self.conv2d.weight
        bias = self.conv2d.bias if self.conv2d.bias is not None else None
        
        # Call Triton kernel
        return triton_conv2d(
            x, weight, bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )