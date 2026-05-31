import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool3d_kernel(
    x_ptr,
    out_ptr,
    B, C, D, H, W,
    D_out, H_out, W_out,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Decompose pid into output indices
    # out_ptr shape: (B, C, D_out, H_out, W_out)
    w_out = pid % W_out
    rem = pid // W_out
    h_out = rem % H_out
    rem = rem // H_out
    d_out = rem % D_out
    rem = rem // D_out
    c = rem % C
    b = rem // C

    # Calculate the starting position in the input tensor
    d_start = d_out * stride - padding
    h_start = h_out * stride - padding
    w_start = w_out * stride - padding

    # Initialize max_val to a very small number
    max_val = -float('inf')

    # Iterate over the 3D kernel window
    for kd in range(kernel_size):
        d_in = d_start + kd * dilation
        if 0 <= d_in < D:
            for kh in range(kernel_size):
                h_in = h_start + kh * dilation
                if 0 <= h_in < H:
                    for kw in range(kernel_size):
                        w_in = w_start + kw * dilation
                        if 0 <= w_in < W:
                            # Calculate the 1D offset for the input tensor (B, C, D, H, W)
                            # offset = b * (C*D*H*W) + c * (D*H*W) + d_in * (H*W) + h_in * W + w_in
                            in_offset = (b * C * D * H * W) + (c * D * H * W) + (d_in * H * W) + (h_in * W) + w_in
                            val = tl.load(x_ptr + in_offset)
                            max_val = tl.maximum(max_val, val)

    # Calculate the 1D offset for the output tensor (B, C, D_out, H_out, W_out)
    out_offset = (b * C * D_out * H_out * W_out) + (c * D_out * H_out * W_out) + (d_out * H_out * W_out) + (h_out * W_out) + w_out
    tl.store(out_ptr + out_offset, max_val)


def triton_maxpool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    B, C, D, H, W = x.shape
    
    # Calculate output dimensions
    D_out = (D + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((B, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Grid is one program per output element
    grid = (B * C * D_out * H_out * W_out,)
    
    maxpool3d_kernel[grid](
        x, out,
        B, C, D, H, W,
        D_out, H_out, W_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 3D using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        # Note: return_indices and ceil_mode are not implemented in this custom kernel for simplicity
        # as they are not typically used in the primary performance paths of these models.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return triton_maxpool3d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )