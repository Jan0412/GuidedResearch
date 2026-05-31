import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B, C_in, H_in, W_in,
    C_out, H_out, W_out,
    stride, padding, dilation, kernel_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Each program handles a block of output elements
    pid = tl.program_id(axis=0)
    n_elements = B * C_out * H_out * W_out
    
    # Calculate output coordinates
    idx = pid
    w_out = idx % W_out
    idx //= W_out
    h_out = idx % H_out
    idx //= H_out
    c_out = idx % C_out
    n = idx // C_out
    
    # Initialize accumulator
    acc = 0.0
    
    # Iterate over input channels and kernel spatial dimensions
    for c_in in range(C_in):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute input coordinates
                h_in = h_out // stride - padding + kh * dilation
                w_in = w_out // stride - padding + kw * dilation
                
                # Check bounds
                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                    # Load weight
                    w_idx = c_out * C_in * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw
                    w = tl.load(w_ptr + w_idx)
                    
                    # Load input
                    x_idx = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in
                    x = tl.load(x_ptr + x_idx)
                    
                    acc += w * x
    
    # Add bias
    if b_ptr is not None:
        b = tl.load(b_ptr + c_out)
        acc += b
    
    # Store output
    out_idx = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out
    tl.store(out_ptr + out_idx, acc)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride: int = 1, padding: int = 0, dilation: int = 1, kernel_size: int = 3):
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get shapes
    B, C_in, H_in, W_in = x.shape
    C_out, _, K, _ = weight.shape
    # Compute output shape
    H_out = (H_in - 1) * stride + 2 * padding - dilation * (kernel_size - 1) - 1 + kernel_size
    W_out = (W_in - 1) * stride + 2 * padding - dilation * (kernel_size - 1) - 1 + kernel_size
    
    # Prepare output
    out = torch.empty(B, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Grid size
    n_elements = B * C_out * H_out * W_out
    
    # Launch kernel
    BLOCK_SIZE_H = 1
    BLOCK_SIZE_W = 1
    
    conv_transpose2d_kernel[(n_elements,)](
        x, weight, bias, out,
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        stride, padding, dilation, kernel_size,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and square kernel, supporting dilation, padding, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square, e.g., 3 for a 3x3 kernel).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in). 

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias if self.conv_transpose2d.bias is not None else None
        
        return triton_conv_transpose2d(
            x, weight, bias,
            stride=self.conv_transpose2d.stride[0],
            padding=self.conv_transpose2d.padding[0],
            dilation=self.conv_transpose2d.dilation[0],
            kernel_size=self.conv_transpose2d.kernel_size[0]
        )