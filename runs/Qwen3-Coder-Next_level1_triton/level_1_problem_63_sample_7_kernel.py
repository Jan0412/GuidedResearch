import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    # Pointers to tensors
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Tensor dimensions
    batch_size, in_channels, out_channels,
    height, width,
    kernel_size,
    stride, padding, dilation,
    # Output dimensions
    out_h, out_w,
    # Block sizes
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # output height index
    pid_c = tl.program_id(2)  # output channel index

    # Compute output position
    out_y = pid_h * BLOCK_H
    out_x = pid_c * BLOCK_W
    
    # Compute the starting position in the input
    in_y_start = out_y * stride - padding
    in_x_start = out_x * stride - padding
    
    # Create output offsets
    output_offsets = (
        pid_b * (out_channels * out_h * out_w) +
        pid_c * (out_h * out_w) +
        tl.arange(0, BLOCK_H)[:, None] * out_w +
        tl.arange(0, BLOCK_W)[None, :]
    )
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(0, in_channels, BLOCK_IC):
        # Loop over kernel height
        for kh in range(kernel_size):
            # Loop over kernel width
            for kw in range(kernel_size):
                # Compute input position
                in_y = in_y_start + kh * dilation
                in_x = in_x_start + kw * dilation
                
                # Check bounds for input height
                valid_y = (in_y >= 0) & (in_y < height)
                # Check bounds for input width
                valid_x = (in_x >= 0) & (in_x < width)
                valid = valid_y & valid_x
                
                # Compute input offsets for this kernel position
                if valid:
                    input_offsets = (
                        pid_b * (in_channels * height * width) +
                        (ic + tl.arange(0, BLOCK_IC)[None, :, None, None]) * (height * width) +
                        in_y * width +
                        in_x
                    )
                    input_offsets = tl.reshape(input_offsets, (BLOCK_IC,))
                    
                    # Load input values
                    mask_ic = (ic + tl.arange(0, BLOCK_IC)) < in_channels
                    input_val = tl.load(input_ptr + input_offsets, mask=mask_ic, other=0.0)
                    
                    # Compute weight offsets
                    weight_offsets = (
                        pid_c * (out_channels * kernel_size * kernel_size * in_channels) +
                        kh * (kernel_size * in_channels) +
                        kw * in_channels +
                        (ic + tl.arange(0, BLOCK_IC))
                    )
                    
                    # Load weight values
                    weight_val = tl.load(weight_ptr + weight_offsets, mask=mask_ic, other=0.0)
                    
                    # Accumulate
                    acc += tl.sum(input_val[None, :] * weight_val[:, None], axis=1).reshape((BLOCK_H, BLOCK_W))
    
    # Add bias if available
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_c)
        acc += bias
    
    # Store output
    tl.store(output_ptr + output_offsets, acc.to(tl.float32), mask=(pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None] < out_h) & 
             (pid_c * BLOCK_W + tl.arange(0, BLOCK_W)[None, :] < out_w))


def triton_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    kernel_size = kernel_h  # assuming square kernel
    
    # Compute output dimensions
    out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_OC = 1
    BLOCK_IC = min(16, in_channels)
    
    # Grid dimensions
    grid = (batch_size, (out_h + BLOCK_H - 1) // BLOCK_H, (out_channels + BLOCK_OC - 1) // BLOCK_OC)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels,
        height, width,
        kernel_size,
        stride, padding, dilation,
        out_h, out_w,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC
    )
    
    return output


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Ensure tensor is on GPU and contiguous
        x = x.contiguous()
        self.weight = self.weight.contiguous()
        
        # Call our Triton-based convolution
        return triton_conv2d(x, self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)