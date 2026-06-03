import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias pointer (can be None)
    out_ptr,  # Output tensor pointer
    n_elements,  # Total elements in output
    # Shape parameters
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height_in: tl.constexpr,
    width_in: tl.constexpr,
    height_out: tl.constexpr,
    width_out: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    # Output tensor index
    out_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_idx < n_elements
    
    # Decode output indices to (batch, out_c, out_h, out_w)
    out_w = out_idx % width_out
    out_idx = out_idx // width_out
    out_h = out_idx % height_out
    out_idx = out_idx // height_out
    out_c = out_idx % out_channels
    out_idx = out_idx // out_channels
    batch = out_idx
    
    # Calculate output value
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop through input channels and kernel positions
    for in_c in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate corresponding input position
                in_h = out_h * stride + kh - padding
                in_w = out_w * stride + kw - padding
                
                # Check if within input bounds
                valid = (in_h >= 0) & (in_h < height_in) & (in_w >= 0) & (in_w < width_in)
                
                # Calculate input index
                in_idx = ((batch * in_channels + in_c) * height_in + in_h) * width_in + in_w
                in_mask = in_idx < batch_size * in_channels * height_in * width_in
                
                # Load input and weight values
                x_val = tl.load(x_ptr + in_idx, mask=valid & in_mask, other=0.0)
                w_idx = ((out_c * in_channels + in_c) * kernel_size + kh) * kernel_size + kw
                w_val = tl.load(w_ptr + w_idx, mask=valid, other=0.0)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        b_val = tl.load(b_ptr + out_c, mask=mask)
        acc += b_val
    
    # Store result
    tl.store(out_ptr + out_idx * width_out + out_w, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                           stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1):
    """
    Triton implementation of ConvTranspose2d.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height_in, width_in = x.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight.shape
    assert kernel_size_h == kernel_size_w, "Only square kernels supported"
    kernel_size = kernel_size_w
    
    # Calculate output dimensions
    height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding
    width_out = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)
    
    n_elements = out.numel()
    BLOCK_SIZE = 256
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        n_elements,
        batch_size, in_channels, out_channels,
        height_in, width_in, height_out, width_out,
        kernel_size, stride, padding, output_padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters but don't create the standard conv layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create weight and bias parameters manually
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights (same as default PyTorch initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using custom Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )