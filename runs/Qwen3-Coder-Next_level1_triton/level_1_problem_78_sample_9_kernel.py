import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias tensor pointer (can be None)
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    in_h,  # Input height
    in_w,  # Input width
    out_h,  # Output height
    out_w,  # Output width
    kernel_h,  # Kernel height
    kernel_w,  # Kernel width
    stride_h,  # Stride height
    stride_w,  # Stride width
    pad_h,  # Padding height
    pad_w,  # Padding width
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)  # For batch and output channel
    pid_n = tl.program_id(1)  # For output position
    
    # Compute output position
    out_idx = pid_n
    ow = out_idx % out_w
    oh = (out_idx // out_w) % out_h
    batch = out_idx // (out_w * out_h)
    
    # For simplicity, we'll process one output channel per program for now
    oc = pid_m
    
    # Compute base offsets in input
    ih_start = oh * stride_h - pad_h
    iw_start = ow * stride_w - pad_w
    
    # Accumulator for the convolution
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for ic in range(in_channels):
        for kh in range(kernel_h):
            ih = ih_start + kh
            # Check if input height is valid
            if ih >= 0 and ih < in_h:
                for kw in range(kernel_w):
                    iw = iw_start + kw
                    # Check if input width is valid
                    if iw >= 0 and iw < in_w:
                        # Compute input index
                        in_idx = batch * (in_channels * in_h * in_w) + \
                                ic * (in_h * in_w) + \
                                ih * in_w + iw
                        # Load input value
                        x_val = tl.load(x_ptr + in_idx)
                        
                        # Compute weight index: [ic, oc, kh, kw] for weight layout
                        # PyTorch uses [in_channels, out_channels, kernel_h, kernel_w] for ConvTranspose2d
                        w_idx = ic * (out_channels * kernel_h * kernel_w) + \
                               oc * (kernel_h * kernel_w) + \
                               kh * kernel_w + kw
                        w_val = tl.load(w_ptr + w_idx)
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val
    
    # Store result
    out_idx_final = batch * (out_channels * out_h * out_w) + \
                   oc * (out_h * out_w) + \
                   oh * out_w + ow
    tl.store(out_ptr + out_idx_final, acc)


def triton_conv_transpose2d(x, weight, bias, stride, padding):
    """
    Custom Triton implementation of 2D transposed convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, in_h, in_w)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_h, kernel_w)
        bias: Bias tensor of shape (out_channels,) or None
        stride: Tuple (stride_h, stride_w)
        padding: Tuple (pad_h, pad_w)
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_h, out_w)
    """
    batch_size, in_channels, in_h, in_w = x.shape
    _, out_channels, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    # Calculate output dimensions
    out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h
    out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w
    
    # Prepare output tensor
    out = torch.empty((batch_size, out_channels, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Grid configuration for parallelization
    # We'll parallelize over output positions and output channels
    num_out_positions = batch_size * out_h * out_w
    num_out_channels = out_channels
    
    # Block sizes (tunable parameters)
    BLOCK_SIZE_M = 1  # For output channels
    BLOCK_SIZE_N = 256  # For output positions
    
    # Grid dimensions
    grid = (
        num_out_channels,
        (num_out_positions + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=32,  # Not used but required for kernel signature
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model class using Triton kernels for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        
        # Initialize weights using Xavier initialization
        kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_h, kernel_w))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using our optimized Triton kernel.
        """
        return triton_conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding)