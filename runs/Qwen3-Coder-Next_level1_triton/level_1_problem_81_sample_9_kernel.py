import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr,              # Input tensor pointer (B, C_in, H_in, W_in)
    w_ptr,              # Weight tensor pointer (C_in, C_out, K_h, K_w)
    b_ptr,              # Bias tensor pointer (C_out,) or None
    y_ptr,              # Output tensor pointer (B, C_out, H_out, W_out)
    B, C_in, C_out,     # Batch size, input channels, output channels
    H_in, W_in,         # Input height, width
    H_out, W_out,       # Output height, width
    K_h, K_w,           # Kernel height, width
    stride_h, stride_w, # Stride
    pad_h, pad_w,       # Padding
    dil_h, dil_w,       # Dilation
    C_out_block: tl.constexpr,
    C_in_block: tl.constexpr,
    K_h_block: tl.constexpr,
    K_w_block: tl.constexpr,
):
    # Program IDs for batch, output channel, output height, output width
    batch_id = tl.program_id(0)
    out_c_id = tl.program_id(1)
    out_h_id = tl.program_id(2)
    out_w_id = tl.program_id(3)
    
    # Compute base indices for the output
    out_h = out_h_id
    out_w = out_w_id
    
    # Compute the starting position in the input for this output position
    # For transposed convolution: input_pos = output_pos - kernel_pos + padding
    # More precisely: out_h = in_h * stride_h + kernel_h * dilation_h - padding_h
    
    # Accumulator for the result
    acc = tl.zeros((C_out_block,), dtype=tl.float32)
    
    # Loop over input channels and kernel positions
    for in_c in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                # Compute corresponding input position
                in_h = (out_h + pad_h - kh * dil_h) // stride_h
                in_w = (out_w + pad_w - kw * dil_w) // stride_w
                
                # Check if input position is valid
                if (in_h >= 0 and in_h < H_in and 
                    in_w >= 0 and in_w < W_in and 
                    (out_h + pad_h - kh * dil_h) % stride_h == 0 and
                    (out_w + pad_w - kw * dil_w) % stride_w == 0):
                    
                    # Compute pointer offsets for input
                    x_offset = (batch_id * C_in * H_in * W_in + 
                               in_c * H_in * W_in + 
                               in_h * W_in + in_w)
                    
                    # Compute pointer offsets for weight
                    w_offset = (in_c * C_out * K_h * K_w + 
                               out_c_id * K_h * K_w + 
                               kh * K_w + kw)
                    
                    # Load values
                    x_val = tl.load(x_ptr + x_offset)
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        bias_offset = out_c_id
        bias_val = tl.load(b_ptr + bias_offset)
        acc += bias_val
    
    # Store result
    y_offset = (batch_id * C_out * H_out * W_out + 
               out_c_id * H_out * W_out + 
               out_h * W_out + out_w)
    tl.store(y_ptr + y_offset, acc.to(x_ptr.dtype.element_ty))


def triton_conv_transpose2d(x, weight, bias, stride, padding, dilation):
    """Custom Triton implementation of ConvTranspose2d"""
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = weight.shape
    
    # Calculate output dimensions (matching PyTorch's ConvTranspose2d)
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_h - 1) + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_w - 1) + 1
    
    # Create output tensor
    y = torch.empty(B, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Configure grid for parallelization
    # Grid: (batch, output_channels, output_height, output_width)
    grid = (B, C_out, H_out, W_out)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        K_h, K_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        C_out_block=1,
        C_in_block=1,
        K_h_block=1,
        K_w_block=1,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for ConvTranspose2d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters and initialize weights/bias manually
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weight with proper initialization
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) / 
            (in_channels * kernel_size * kernel_size) ** 0.5
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using custom Triton kernel.
        """
        # Convert stride, padding, dilation to tuples if they are integers
        stride = (self.stride, self.stride)
        padding = (self.padding, self.padding)
        dilation = (self.dilation, self.dilation)
        
        return triton_conv_transpose2d(
            x, self.weight, self.bias, 
            stride, padding, dilation
        )