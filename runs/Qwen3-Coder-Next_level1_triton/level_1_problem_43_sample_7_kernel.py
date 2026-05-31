import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_pool3d_kernel(
    X,  # Pointer to input tensor
    Y,  # Pointer to output tensor
    X_stride0, X_stride1, X_stride2, X_stride3, X_stride4,  # Input strides
    Y_stride0, Y_stride1, Y_stride2, Y_stride3, Y_stride4,  # Output strides
    B, C, D_in, H_in, W_in,  # Input dimensions
    D_out, H_out, W_out,  # Output dimensions
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the batch, channel, and spatial indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    d_out_idx = tl.program_id(2)
    h_out_idx = tl.program_id(3)
    w_out_idx = tl.program_id(4)
    
    # Calculate the starting position in the input for this output element
    d_in_start = d_out_idx * stride - padding
    h_in_start = h_out_idx * stride - padding
    w_in_start = w_out_idx * stride - padding
    
    # Initialize max value to -inf
    max_val = -tl.float32("inf")
    
    # Iterate through the kernel window
    for kd in range(kernel_size):
        d_in = d_in_start + kd * dilation
        # Skip if outside input bounds
        if d_in >= 0 and d_in < D_in:
            for kh in range(kernel_size):
                h_in = h_in_start + kh * dilation
                if h_in >= 0 and h_in < H_in:
                    for kw in range(kernel_size):
                        w_in = w_in_start + kw * dilation
                        if w_in >= 0 and w_in < W_in:
                            # Calculate input pointer offset
                            offset = (batch_idx * X_stride0 +
                                     channel_idx * X_stride1 +
                                     d_in * X_stride2 +
                                     h_in * X_stride3 +
                                     w_in * X_stride4)
                            val = tl.load(X + offset)
                            max_val = tl.maximum(max_val, val)
    
    # Calculate output pointer offset
    offset = (batch_idx * Y_stride0 +
             channel_idx * Y_stride1 +
             d_out_idx * Y_stride2 +
             h_out_idx * Y_stride3 +
             w_out_idx * Y_stride4)
    tl.store(Y + offset, max_val)


def triton_max_pool3d(x, kernel_size, stride, padding, dilation):
    """
    Apply 3D max pooling using Triton kernel.
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get input dimensions
    batch_size, channels, D_in, H_in, W_in = x.shape
    
    # Calculate output dimensions manually (same as PyTorch's MaxPool3d)
    def calc_out_dim(in_dim, k, s, p, d):
        return (in_dim + 2*p - d*(k-1) - 1) // s + 1
    
    D_out = calc_out_dim(D_in, kernel_size, stride, padding, dilation)
    H_out = calc_out_dim(H_in, kernel_size, stride, padding, dilation)
    W_out = calc_out_dim(W_in, kernel_size, stride, padding, dilation)
    
    # Create output tensor
    y = torch.empty(batch_size, channels, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Calculate strides
    X_stride0, X_stride1, X_stride2, X_stride3, X_stride4 = x.stride()
    Y_stride0, Y_stride1, Y_stride2, Y_stride3, Y_stride4 = y.stride()
    
    # Grid dimensions: (batch_size, channels, D_out, H_out, W_out)
    grid = (batch_size, channels, D_out, H_out, W_out)
    
    # Launch the kernel
    max_pool3d_kernel[grid](
        x, y,
        X_stride0, X_stride1, X_stride2, X_stride3, X_stride4,
        Y_stride0, Y_stride1, Y_stride2, Y_stride3, Y_stride4,
        batch_size, channels, D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        BLOCK_SIZE=32
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 3D using custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
        # Validate parameters that Triton kernel doesn't support yet
        if self.return_indices:
            raise ValueError("return_indices=True is not supported in the Triton implementation")
        if self.ceil_mode:
            raise ValueError("ceil_mode=True is not supported in the Triton implementation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return triton_max_pool3d(x, self.kernel_size, self.stride, self.padding, self.dilation)