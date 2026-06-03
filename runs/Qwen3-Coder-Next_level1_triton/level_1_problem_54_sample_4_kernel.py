import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,  # Input tensor pointer (B, C_in, D, H, W)
    weight_ptr,  # Weight tensor pointer (C_out, C_in, Kd, Kh, Kw)
    bias_ptr,  # Bias tensor pointer (C_out,)
    output_ptr,  # Output tensor pointer (B, C_out, D_out, H_out, W_out)
    batch_size,  # B
    in_channels,  # C_in
    out_channels,  # C_out
    depth, height, width,  # Input dimensions
    out_depth, out_height, out_width,  # Output dimensions
    kernel_size_d, kernel_size_h, kernel_size_w,  # Kernel dimensions
    stride_d, stride_h, stride_w,  # Strides
    pad_d, pad_h, pad_w,  # Padding
    dilation_d, dilation_h, dilation_w,  # Dilations
    BLOCK_SIZE: tl.constexpr,
):
    # Get output tensor indices
    pid = tl.program_id(0)
    
    # Calculate indices for the output tensor
    # Output shape: (B, C_out, D_out, H_out, W_out)
    total_elements = batch_size * out_channels * out_depth * out_height * out_width
    if pid >= total_elements:
        return
    
    # Decode linear index to (b, oc, od, oh, ow)
    temp = pid
    ow = temp % out_width
    temp //= out_width
    oh = temp % out_height
    temp //= out_height
    od = temp % out_depth
    temp //= out_depth
    oc = temp % out_channels
    b = temp // out_channels
    
    # Calculate the starting position in input space
    input_d = od * stride_d - pad_d + tl.arange(0, kernel_size_d) * dilation_d
    input_h = oh * stride_h - pad_h + tl.arange(0, kernel_size_h) * dilation_h
    input_w = ow * stride_w - pad_w + tl.arange(0, kernel_size_w) * dilation_w
    
    # Accumulate convolution result
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over input channels
    for ic in range(in_channels):
        # Iterate over kernel dimensions
        for kd in range(kernel_size_d):
            for kh in range(kernel_size_h):
                for kw in range(kernel_size_w):
                    d_idx = input_d[kd]
                    h_idx = input_h[kh]
                    w_idx = input_w[kw]
                    
                    # Check if indices are valid (handle padding)
                    valid_mask = (d_idx >= 0) & (d_idx < depth) & \
                                (h_idx >= 0) & (h_idx < height) & \
                                (w_idx >= 0) & (w_idx < width)
                    
                    # Load input values
                    input_offset = b * (in_channels * depth * height * width) + \
                                  ic * (depth * height * width) + \
                                  d_idx * (height * width) + \
                                  h_idx * width + w_idx
                    input_val = tl.load(input_ptr + input_offset, mask=valid_mask, other=0.0)
                    
                    # Load weight values
                    weight_offset = oc * (in_channels * kernel_size_d * kernel_size_h * kernel_size_w) + \
                                   ic * (kernel_size_d * kernel_size_h * kernel_size_w) + \
                                   kd * (kernel_size_h * kernel_size_w) + \
                                   kh * kernel_size_w + kw
                    weight_val = tl.load(weight_ptr + weight_offset)
                    
                    # Accumulate multiplication
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_offset = oc
        acc += tl.load(bias_ptr + bias_offset)
    
    # Store result
    output_offset = pid
    tl.store(output_ptr + output_offset, acc.to(tl.float32))


def triton_conv3d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Triton-based 3D convolution implementation.
    
    Args:
        input_tensor: Input tensor of shape (batch_size, in_channels, depth, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding applied to input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections (must be 1 for this implementation)
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, depth, height, width = input_tensor.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_depth = (depth + 2 * padding - dilation * (kernel_d - 1) - 1) // stride + 1
    out_height = (height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Grid configuration
    total_elements = batch_size * out_channels * out_depth * out_height * out_width
    BLOCK_SIZE = 128
    grid = (total_elements,)
    
    # Launch kernel
    conv3d_kernel[grid](
        input_tensor,
        weight,
        bias,
        output,
        batch_size,
        in_channels,
        out_channels,
        depth, height, width,
        out_depth, out_height, out_width,
        kernel_d, kernel_h, kernel_w,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized 3D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Register parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters (same as nn.Conv3d)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        return triton_conv3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )