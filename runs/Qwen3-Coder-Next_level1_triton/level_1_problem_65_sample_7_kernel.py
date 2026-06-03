import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,            # Input tensor pointer
    w_ptr,            # Weight tensor pointer
    b_ptr,            # Bias tensor pointer (optional)
    y_ptr,            # Output tensor pointer
    batch_size,       # B
    in_channels,      # C_in
    out_channels,     # C_out
    in_h,             # Input height
    in_w,             # Input width
    out_h,            # Output height
    out_w,            # Output width
    kernel_h,         # Kernel height
    kernel_w,         # Kernel width
    stride,           # Stride
    padding,          # Padding
    output_padding,   # Output padding
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output position
    out_idx = tl.program_id(0)
    batch_idx = out_idx // (out_channels * out_h * out_w)
    out_idx = out_idx % (out_channels * out_h * out_w)
    out_c = out_idx // (out_h * out_w)
    out_idx = out_idx % (out_h * out_w)
    out_row = out_idx // out_w
    out_col = out_idx % out_w
    
    # Compute starting position in input
    in_row_start = out_row - kernel_h + 1 + padding
    in_col_start = out_col - kernel_w + 1 + padding
    
    # Check if this output position can be computed
    if batch_idx >= batch_size:
        return
    
    # Accumulator for the result
    acc = 0.0
    if b_ptr is not None:
        acc = tl.load(b_ptr + out_c)
    
    # Iterate over the kernel and input
    for in_c in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Compute corresponding input position
                in_row = in_row_start + kh * stride
                in_col = in_col_start + kw * stride
                
                # Check bounds
                if (in_row >= 0 and in_row < in_h and 
                    in_col >= 0 and in_col < in_w):
                    # Compute indices
                    x_idx = (batch_idx * in_channels * in_h * in_w +
                            in_c * in_h * in_w +
                            in_row * in_w + in_col)
                    w_idx = (out_c * in_channels * kernel_h * kernel_w +
                            in_c * kernel_h * kernel_w +
                            kh * kernel_w + kw)
                    
                    # Load values and accumulate
                    x_val = tl.load(x_ptr + x_idx)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val
    
    # Store result
    y_idx = out_idx + batch_idx * out_channels * out_h * out_w
    tl.store(y_ptr + y_idx, acc)


def triton_conv_transpose2d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose2d.
    
    Args:
        x: Input tensor [B, C_in, H_in, W_in]
        weight: Weight tensor [C_in, C_out, K_h, K_w] (PyTorch stores as [C_out, C_in, K_h, K_w])
        bias: Optional bias tensor [C_out]
        stride, padding, output_padding, groups: Convolution parameters
    
    Returns:
        Output tensor [B, C_out, H_out, W_out]
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Extract shapes
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, in_channels_w, kernel_h, kernel_w = weight.shape
    
    assert in_channels == in_channels_w, "Input channels must match"
    assert groups == 1, "Groups > 1 not supported in this kernel"
    
    # Compute output dimensions
    out_h = (in_h - 1) * stride - 2 * padding + output_padding + kernel_h
    out_w = (in_w - 1) * stride - 2 * padding + output_padding + kernel_w
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Compute total number of output elements
    n_elements = batch_size * out_channels * out_h * out_w
    
    # Configure kernel launch
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
        stride, padding, output_padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for ConvTranspose2d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same convolution layer
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding, 
            groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Triton kernel for convolution.
        """
        # Get parameters from the original layer
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias
        
        # Call our optimized Triton implementation
        return triton_conv_transpose2d(
            x, weight, bias,
            stride=self.conv_transpose2d.stride,
            padding=self.conv_transpose2d.padding,
            output_padding=self.conv_transpose2d.output_padding,
            groups=self.conv_transpose2d.groups
        )