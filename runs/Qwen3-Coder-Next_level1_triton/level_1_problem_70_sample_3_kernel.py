import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, depth, height, width)
    w_ptr,  # Weight tensor: (in_channels, out_channels, k_d, k_h, k_w)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, out_d, out_h, out_w)
    batch_size, in_channels, out_channels,
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    k_d, k_h, k_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    output_padding_d, output_padding_h, output_padding_w,
    dilation_d, dilation_h, dilation_w,
    n_blocks_d: tl.constexpr,
    n_blocks_h: tl.constexpr,
    n_blocks_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get batch, out_channel indices
    bc = tl.program_id(0)
    batch_idx = bc // out_channels
    out_channel_idx = bc % out_channels
    
    # Calculate 3D position in output tensor
    # We'll parallelize over output positions using 3D grid
    out_d_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Compute output pointer offset
    out_offset = (batch_idx * out_channels * out_d * out_h * out_w +
                  out_channel_idx * out_d * out_h * out_w +
                  out_d_idx * out_h * out_w +
                  out_h_idx * out_w +
                  out_w_idx)
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels
    for in_c in range(in_channels):
        # Compute input position corresponding to this output position
        input_d = out_d_idx * stride_d - padding_d + in_c * 0  # Will be computed per kernel element
        input_h = out_h_idx * stride_h - padding_h
        input_w = out_w_idx * stride_w - padding_w
        
        # Loop over kernel dimensions
        for kd in range(k_d):
            for kh in range(k_h):
                for kw in range(k_w):
                    # Compute actual input position
                    in_d_pos = out_d_idx * stride_d - padding_d + kd * dilation_d
                    in_h_pos = out_h_idx * stride_h - padding_h + kh * dilation_h
                    in_w_pos = out_w_idx * stride_w - padding_w + kw * dilation_w
                    
                    # Check if input position is valid
                    if (0 <= in_d_pos < in_d and 
                        0 <= in_h_pos < in_h and 
                        0 <= in_w_pos < in_w):
                        # Compute input offset
                        in_offset = (batch_idx * in_channels * in_d * in_h * in_w +
                                    in_c * in_d * in_h * in_w +
                                    in_d_pos * in_h * in_w +
                                    in_h_pos * in_w +
                                    in_w_pos)
                        
                        # Compute weight offset
                        w_offset = (in_c * out_channels * k_d * k_h * k_w +
                                   out_channel_idx * k_d * k_h * k_w +
                                   kd * k_h * k_w +
                                   kh * k_w +
                                   kw)
                        
                        # Load values
                        x_val = tl.load(x_ptr + in_offset)
                        w_val = tl.load(w_ptr + w_offset)
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_offset = out_channel_idx
        acc += tl.load(b_ptr + b_offset)
    
    # Store result
    tl.store(out_ptr + out_offset, acc)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Triton implementation of ConvTranspose3d
    
    Args:
        x: Input tensor of shape (batch, in_channels, depth, height, width)
        weight: Weight tensor of shape (in_channels, out_channels, k_d, k_h, k_w)
        bias: Optional bias tensor of shape (out_channels,)
        Other parameters: same as torch.nn.ConvTranspose3d
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, in_d, in_h, in_w = x.shape
    _, out_channels, k_d, k_h, k_w = weight.shape
    
    # Compute output dimensions (same logic as PyTorch)
    stride_d = stride_h = stride_w = stride if isinstance(stride, int) else stride
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride
        
    dilation_d = dilation_h = dilation_w = dilation if isinstance(dilation, int) else dilation
    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation
        
    padding_d = padding_h = padding_w = padding if isinstance(padding, int) else padding
    if isinstance(padding, int):
        padding_d = padding_h = padding_w = padding
    else:
        padding_d, padding_h, padding_w = padding
        
    output_padding_d = output_padding_h = output_padding_w = output_padding if isinstance(output_padding, int) else output_padding
    if isinstance(output_padding, int):
        output_padding_d = output_padding_h = output_padding_w = output_padding
    else:
        output_padding_d, output_padding_h, output_padding_w = output_padding
    
    # Calculate output dimensions
    out_d = (in_d - 1) * stride_d - 2 * padding_d + dilation_d * (k_d - 1) + output_padding_d + 1
    out_h = (in_h - 1) * stride_h - 2 * padding_h + dilation_h * (k_h - 1) + output_padding_h + 1
    out_w = (in_w - 1) * stride_w - 2 * padding_w + dilation_w * (k_w - 1) + output_padding_w + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Launch configuration
    # Program 0: batch * out_channels
    # Program 1: out_d
    # Program 2: out_h
    # Program 3: out_w
    grid = (batch_size * out_channels, out_d, out_h, out_w)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        k_d, k_h, k_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        n_blocks_d=1, n_blocks_h=1, n_blocks_w=1,
        BLOCK_SIZE=1
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for ConvTranspose3d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create the same layer structure but replace the forward pass with our Triton kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )