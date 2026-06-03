import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor pointer: (batch, in_channels, H_in, W_in)
    w_ptr,  # Weight tensor pointer: (in_channels, out_channels, kH, kW)
    b_ptr,  # Bias tensor pointer: (out_channels,) or nullptr
    out_ptr,  # Output tensor pointer: (batch, out_channels, H_out, W_out)
    batch_size, in_channels, out_channels,
    height_in, width_in,
    kernel_size, stride, padding, dilation,
    H_out, W_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output positions
    out_h = pid_h
    out_w = pid_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels and kernel positions
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate corresponding input position
                in_h = (out_h - (kh * dilation + padding)) // stride
                in_w = (out_w - (kw * dilation + padding)) // stride
                
                # Check if input position is valid
                valid = (in_h >= 0) & (in_h < height_in) & (in_w >= 0) & (in_w < width_in)
                
                if valid:
                    # Calculate input pointer offset
                    in_offset = (pid_batch * in_channels * height_in * width_in +
                                ic * height_in * width_in +
                                in_h * width_in + in_w)
                    x_val = tl.load(x_ptr + in_offset)
                    
                    # Calculate weight pointer offset
                    w_offset = (ic * out_channels * kernel_size * kernel_size +
                               pid_out_c * kernel_size * kernel_size +
                               kh * kernel_size + kw)
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_offset = pid_out_c
        bias_val = tl.load(b_ptr + bias_offset)
        acc += bias_val
    
    # Store result
    out_offset = (pid_batch * out_channels * H_out * W_out +
                 pid_out_c * H_out * W_out +
                 out_h * W_out + out_w)
    tl.store(out_ptr + out_offset, acc)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0, dilation: int = 1):
    """
    Performs 2D transposed convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, H_in, W_in)
        weight: Weight tensor of shape (in_channels, out_channels, kH, kW)
        bias: Optional bias tensor of shape (out_channels,)
        stride, padding, dilation: convolution parameters
    
    Returns:
        Output tensor of shape (batch_size, out_channels, H_out, W_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height_in, width_in = x.shape
    out_channels = weight.shape[1]
    kernel_size = weight.shape[2]
    
    # Calculate output dimensions
    H_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    W_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Define block sizes for optimization
    BLOCK_SIZE_M = 1  # We process one output element per thread block
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 1
    
    # Grid dimensions
    grid = (batch_size, out_channels, H_out, W_out)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        height_in, width_in,
        kernel_size, stride, padding, dilation,
        H_out, W_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with custom Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias similar to nn.ConvTranspose2d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights using Kaiming uniform initialization (similar to PyTorch default)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and bias similar to PyTorch's ConvTranspose2d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )