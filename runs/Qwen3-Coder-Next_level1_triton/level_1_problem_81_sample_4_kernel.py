import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor (batch, in_channels, H_in, W_in)
    w_ptr,  # Weight tensor (in_channels, out_channels, kH, kW)
    b_ptr,  # Bias tensor (out_channels) [optional]
    y_ptr,  # Output tensor (batch, out_channels, H_out, W_out)
    batch_size, in_channels, out_channels,
    height_in, width_in,
    height_out, width_out,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch * out_channels dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for in_channels dimension
    BLOCK_SIZE_KH: tl.constexpr,  # Block size for kernel height
    BLOCK_SIZE_KW: tl.constexpr,  # Block size for kernel width
):
    # Get program IDs
    pid_batch_out = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Compute batch and out_channel indices from pid_batch_out
    pid_batch = pid_batch_out // out_channels
    pid_out_channel = pid_batch_out % out_channels
    
    # Check bounds
    if pid_batch >= batch_size or pid_h >= height_out or pid_w >= width_out:
        return
    
    # Compute output position
    y_ptr += pid_batch * (out_channels * height_out * width_out) + \
             pid_out_channel * (height_out * width_out) + \
             pid_h * width_out + pid_w
    
    # Accumulator for the result
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(in_channels):
        # Compute input position for this channel
        # For transposed convolution: output position (ph, pw) gets contributions from input positions
        # where (ph - k*stride + pad) % dilation == 0 and similarly for width
        
        # Iterate over kernel height
        for kh in range(0, kernel_size, BLOCK_SIZE_KH):
            kh_end = tl.minimum(kh + BLOCK_SIZE_KH, kernel_size)
            for kw in range(0, kernel_size, BLOCK_SIZE_KW):
                kw_end = tl.minimum(kw + BLOCK_SIZE_KW, kernel_size)
                
                # Compute input position that would contribute to (pid_h, pid_w)
                # h_in = (pid_h - (kh - (kernel_size-1)/2) * dilation - padding) / stride
                # But we need to handle the actual mapping for transposed conv
                
                # For each kernel element (kh, kw), compute which input position contributes
                # In transposed conv: y[batch, out_c, h_out, w_out] += 
                #   x[batch, in_c, h_in, w_in] * w[in_c, out_c, kh, kw]
                # where h_in = (h_out - kh*dilation) // stride + padding//stride ? 
                
                # Actually, standard transposed conv mapping:
                # h_in = (pid_h + padding - kh * dilation) // stride
                # w_in = (pid_w + padding - kw * dilation) // stride
                
                h_in = (pid_h + padding - kh * dilation) // stride
                w_in = (pid_w + padding - kw * dilation) // stride
                
                # Check if h_in, w_in are valid
                if h_in >= 0 and h_in < height_in and w_in >= 0 and w_in < width_in:
                    # Check if the division is exact (stride must align)
                    if (pid_h + padding - kh * dilation) % stride == 0 and \
                       (pid_w + padding - kw * dilation) % stride == 0:
                        
                        # Compute input pointer offset
                        x_offset = pid_batch * (in_channels * height_in * width_in) + \
                                  c * (height_in * width_in) + \
                                  h_in * width_in + w_in
                        x_val = tl.load(x_ptr + x_offset)
                        
                        # Compute weight pointer offset
                        w_offset = c * (out_channels * kernel_size * kernel_size) + \
                                  pid_out_channel * (kernel_size * kernel_size) + \
                                  kh * kernel_size + kw
                        w_val = tl.load(w_ptr + w_offset)
                        
                        acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_channel)
        acc += bias
    
    # Store result
    tl.store(y_ptr, acc.to(y_ptr.dtype.element_ty))


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Triton implementation of 2D transposed convolution.
    """
    batch_size, in_channels, height_in, width_in = x.shape
    in_channels_w, out_channels, kernel_size_h, kernel_size_w = weight.shape
    
    assert kernel_size_h == kernel_size_w, "Only square kernels supported"
    kernel_size = kernel_size_h
    
    # Compute output dimensions
    height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    y = torch.empty(batch_size, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)
    
    # Check if bias is provided
    has_bias = bias is not None
    
    # Configure grid
    # Each block computes one (batch, out_channel, h_out, w_out) position
    grid = lambda meta: (
        batch_size * out_channels,
        height_out,
        width_out
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias if has_bias else None, y,
        batch_size, in_channels, out_channels,
        height_in, width_in,
        height_out, width_out,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=16, BLOCK_SIZE_KH=4, BLOCK_SIZE_KW=4
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 
            (2.0 / (kernel_size * kernel_size * in_channels)) ** 0.5
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )