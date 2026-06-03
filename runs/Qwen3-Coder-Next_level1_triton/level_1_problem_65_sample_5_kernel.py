import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    X,  # Input tensor (B, C_in, H, W)
    W,  # Weight tensor (C_in, C_out, K_h, K_w)
    B,  # Bias tensor (C_out,)
    Y,  # Output tensor (B, C_out, H_out, W_out)
    B_in, C_in, H_in, W_in,  # Input dimensions
    B_out, C_out, H_out, W_out,  # Output dimensions
    K_h, K_w,  # Kernel dimensions
    stride, padding, output_padding,
    # Meta-parameters
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs for batch, output channels, and spatial coordinates
    batch_id = tl.program_id(0)
    c_out_id = tl.program_id(1)
    h_out_id = tl.program_id(2)
    w_out_id = tl.program_id(3)
    
    # Calculate output pointers
    y_ptr = Y + batch_id * C_out * H_out * W_out + c_out_id * H_out * W_out + h_out_id * W_out + w_out_id
    
    # Initialize accumulator
    acc = 0.0
    
    # Iterate over input channels and kernel positions
    for c_in in range(C_in):
        for kh in range(K_h):
            # Calculate corresponding input position
            h_in = h_out_id + kh - padding - output_padding // 2
            # Check if h_in is within valid input range
            if h_in >= 0 and h_in < H_in:
                for kw in range(K_w):
                    w_in = w_out_id + kw - padding - output_padding // 2
                    # Check if w_in is within valid input range
                    if w_in >= 0 and w_in < W_in:
                        # Calculate pointers for input and weight
                        x_ptr = X + batch_id * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in
                        w_ptr = W + c_in * C_out * K_h * K_w + c_out_id * K_h * K_w + kh * K_w + kw
                        # Accumulate the product
                        acc += tl.load(x_ptr) * tl.load(w_ptr)
    
    # Add bias if present
    if B is not None:
        acc += tl.load(B + c_out_id)
    
    # Store the result
    tl.store(y_ptr, acc)

def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Performs transposed 2D convolution using Triton kernel.
    """
    # Get dimensions
    B, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = weight.shape
    
    # Calculate output dimensions (same as PyTorch's ConvTranspose2d)
    H_out = (H_in - 1) * stride - 2 * padding + K_h + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + K_w + output_padding
    
    # Create output tensor
    y = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Configure grid: (batch, output_channels, output_height, output_width)
    # For efficiency, we'll parallelize over batch, output channels, and combine spatial dimensions
    grid = (B, C_out, H_out, W_out)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, y,
        B, C_in, H_in, W_in,
        B, C_out, H_out, W_out,
        K_h, K_w,
        stride, padding, output_padding,
        BLOCK_SIZE_C_OUT=1,
        BLOCK_SIZE_C_IN=1,
        BLOCK_SIZE_H=1,
        BLOCK_SIZE_W=1,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of the transposed 2D convolution with custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create weight and bias parameters similar to nn.ConvTranspose2d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.has_bias = bias
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Initialize weight and bias similar to PyTorch's ConvTranspose2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.has_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None
        
        # Call our optimized Triton kernel
        return triton_conv_transpose2d(
            x, weight, bias,
            stride=self.stride,
            padding=self.padding[0],  # Using first padding value for simplicity
            output_padding=self.output_padding[0],  # Using first output_padding value
            groups=self.groups
        )
    
    def extra_repr(self):
        return '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, output_padding={output_padding}, groups={groups}, bias={has_bias}'.format(**self.__dict__)