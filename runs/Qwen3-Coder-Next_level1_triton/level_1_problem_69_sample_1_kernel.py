import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, h, w)
    w_ptr,  # Weight tensor: (in_channels, out_channels, k_h, k_w)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, h_out, w_out)
    batch_size, in_channels, out_channels,
    h_in, w_in, h_out, w_out,
    k_h, k_w, stride_h, stride_w, padding_h, padding_w, output_padding_h, output_padding_w, dilation_h, dilation_w,
    n_elements,  # Total elements in output
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Calculate output tensor index
    out_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_idx < n_elements
    
    # Decode linear index to (b, oc, oh, ow)
    ow = out_idx % w_out
    tmp = out_idx // w_out
    oh = tmp % h_out
    tmp = tmp // h_out
    oc = tmp % out_channels
    b = tmp // out_channels
    
    # Accumulator for the output
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for ic in range(in_channels):
        for kh in range(k_h):
            for kw in range(k_w):
                # Compute input position corresponding to this kernel position
                h_in_pos = (oh - padding_h + kh * dilation_h) // stride_h
                w_in_pos = (ow - padding_w + kw * dilation_w) // stride_w
                
                # Check if this position is valid for input
                valid_pos = (h_in_pos >= 0) & (h_in_pos < h_in) & \
                           (w_in_pos >= 0) & (w_in_pos < w_in) & \
                           ((oh - padding_h + kh * dilation_h) % stride_h == 0) & \
                           ((ow - padding_w + kw * dilation_w) % stride_w == 0)
                
                # Calculate input index
                input_idx = b * (in_channels * h_in * w_in) + \
                           ic * (h_in * w_in) + \
                           h_in_pos * w_in + w_in_pos
                
                # Load input value
                x_val = tl.load(x_ptr + input_idx, mask=valid_pos, other=0.0)
                
                # Calculate weight index: weight[ic, oc, kh, kw]
                weight_idx = (ic * out_channels * k_h * k_w + 
                             oc * k_h * k_w + 
                             kh * k_w + kw)
                w_val = tl.load(w_ptr + weight_idx)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if present
    if HAS_BIAS:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val
    
    # Store result
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=mask)


def triton_conv_transpose2d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    """
    Triton implementation of ConvTranspose2d for groups=1
    """
    # Extract parameters
    batch_size, in_channels, h_in, w_in = x.shape
    out_channels, _, k_h, k_w = weight.shape
    
    # Calculate output dimensions
    h_out = (h_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (k_h - 1) + output_padding[0] + 1
    w_out = (w_in - 1) * stride[1] - 2 * padding[1] - 2 * padding[1] + dilation[1] * (k_w - 1) + output_padding[1] + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, h_out, w_out, dtype=x.dtype, device=x.device)
    n_elements = out.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias if bias is not None else None, out,
        batch_size, in_channels, out_channels,
        h_in, w_in, h_out, w_out,
        k_h, k_w, stride[0], stride[1], padding[0], padding[1], output_padding[0], output_padding[1], dilation[0], dilation[1],
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_BIAS=bias is not None,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.
    Uses optimized Triton kernel for the convolution operation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), output_padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters (same as nn.ConvTranspose2d)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using optimized Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.dilation, self.groups
        )