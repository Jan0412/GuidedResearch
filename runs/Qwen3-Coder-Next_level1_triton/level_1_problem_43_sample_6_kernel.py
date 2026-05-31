import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr,  # Input tensor pointer (batch, channels, d, h, w)
    y_ptr,  # Output tensor pointer
    n_elements,  # Total number of output elements
    # Input tensor dimensions
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    input_d: tl.constexpr,
    input_h: tl.constexpr,
    input_w: tl.constexpr,
    # Output tensor dimensions
    output_d: tl.constexpr,
    output_h: tl.constexpr,
    output_w: tl.constexpr,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_d: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    # Block size for parallelization
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the global index for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Convert 1D offset to 5D output indices: (batch, channel, out_d, out_h, out_w)
    # Output dimensions: [batch_size, channels, output_d, output_h, output_w]
    
    # Calculate output indices from linear offset
    # offset = (((batch * channels + channel) * output_d + out_d) * output_h + out_h) * output_w + out_w
    
    temp = offsets
    out_w = temp % output_w
    temp = temp // output_w
    out_h = temp % output_h
    temp = temp // output_h
    out_d = temp % output_d
    temp = temp // output_d
    channel = temp % channels
    batch = temp // channels
    
    # Compute input starting positions (top-left-front corner of the pooling window)
    in_d_start = out_d * stride_d - pad_d
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Initialize max value with very small number
    max_val = -float("inf")
    
    # Iterate over the pooling window
    for kd in range(kernel_d):
        in_d = in_d_start + kd * dil_d
        for kh in range(kernel_h):
            in_h = in_h_start + kh * dil_h
            for kw in range(kernel_w):
                in_w = in_w_start + kw * dil_w
                
                # Check if within bounds (handle padding)
                valid_d = (in_d >= 0) & (in_d < input_d)
                valid_h = (in_h >= 0) & (in_h < input_h)
                valid_w = (in_w >= 0) & (in_w < input_w)
                valid = valid_d & valid_h & valid_w
                
                # Calculate input pointer offset for this position
                # Input layout: batch, channel, d, h, w
                # offset = (((batch * channels + channel) * input_d + in_d) * input_h + in_h) * input_w + in_w
                input_offset = (((batch * channels + channel) * input_d + in_d) * input_h + in_h) * input_w + in_w
                
                # Load value only if valid position
                x_val = tl.load(x_ptr + input_offset, mask=valid, other=-float("inf"))
                max_val = tl.maximum(max_val, x_val)
    
    # Store the result
    tl.store(y_ptr + offsets, max_val, mask=mask)


def triton_maxpool3d(x, kernel_size, stride, padding, dilation, output_size):
    """
    Triton implementation of 3D max pooling.
    
    Args:
        x: Input tensor of shape (batch_size, channels, dim1, dim2, dim3)
        kernel_size: Pooling kernel size (int or tuple)
        stride: Stride (int or tuple)
        padding: Padding (int or tuple)
        dilation: Dilation (int or tuple)
        output_size: Precomputed output size (d, h, w)
    
    Returns:
        Output tensor with max pooling applied
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Extract parameters
    batch_size, channels, input_d, input_h, input_w = x.shape
    output_d, output_h, output_w = output_size
    
    # Convert parameters to tuple if int
    if isinstance(kernel_size, int):
        kernel_d = kernel_h = kernel_w = kernel_size
    else:
        kernel_d, kernel_h, kernel_w = kernel_size
        
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_d = pad_h = pad_w = padding
    else:
        pad_d, pad_h, pad_w = padding
        
    if isinstance(dilation, int):
        dil_d = dil_h = dil_w = dilation
    else:
        dil_d, dil_h, dil_w = dilation
        
    # Create output tensor
    out = torch.empty(batch_size, channels, output_d, output_h, output_w, dtype=x.dtype, device=x.device)
    
    # Total number of output elements
    n_elements = batch_size * channels * output_d * output_h * output_w
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Compute grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    maxpool3d_kernel[grid](
        x, out, n_elements,
        batch_size, channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Max Pooling 3D model using Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the optimized Max Pooling 3D layer.

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
        
        # Store parameters for output size calculation
        self.register_buffer('_kernel_size', torch.tensor(kernel_size))
        self.register_buffer('_stride', torch.tensor(self.stride))
        self.register_buffer('_padding', torch.tensor(padding))
        self.register_buffer('_dilation', torch.tensor(dilation))
        self.register_buffer('_ceil_mode', torch.tensor(1 if ceil_mode else 0))
        
        if return_indices:
            raise NotImplementedError("return_indices=True is not supported in this Triton implementation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        # Calculate output size
        batch_size, channels, input_d, input_h, input_w = x.shape
        
        # Calculate output dimensions
        if self.ceil_mode:
            output_d = int(torch.ceil(torch.tensor((input_d + 2 * self.padding - self.dilation * (self._kernel_size - 1) - 1) / self._stride + 1)).item())
            output_h = int(torch.ceil(torch.tensor((input_h + 2 * self.padding - self.dilation * (self._kernel_size - 1) - 1) / self._stride + 1)).item())
            output_w = int(torch.ceil(torch.tensor((input_w + 2 * self.padding - self.dilation * (self._kernel_size - 1) - 1) / self._stride + 1)).item())
        else:
            output_d = int(torch.floor(torch.tensor((input_d + 2 * self.padding - self.dilation * (self._kernel_size - 1) - 1) / self._stride + 1)).item())
            output_h = int(torch.floor(torch.tensor((input_h + 2 * self.padding - self.dilation * (self._kernel_size - 1) - 1) / self._stride + 1)).item())
            output_w = int(torch.floor(torch.tensor((input_w + 2 * self.padding - self.dilation * (self._kernel_size - 1) - 1) / self._stride + 1)).item())
        
        # Handle negative output dimensions (when input is too small)
        output_d = max(1, output_d)
        output_h = max(1, output_h)
        output_w = max(1, output_w)
        
        # Call Triton implementation
        return triton_maxpool3d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation, 
            (output_d, output_h, output_w)
        )