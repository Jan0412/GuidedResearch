import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias tensor pointer (optional)
    out_ptr,  # Output tensor pointer
    batch_size, in_channels, out_channels,
    input_w, input_h, input_d,
    kernel_w, kernel_h, kernel_d,
    stride_w, stride_h, stride_d,
    pad_w, pad_h, pad_d,
    dil_w, dil_h, dil_d,
    output_w, output_h, output_d,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for computation
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Calculate output position
    out_h_idx = pid_h
    out_w_idx = tl.program_id(3) if tl.constexpr(tl.num_programs(3) > 1) else 0
    out_d_idx = tl.program_id(4) if tl.constexpr(tl.num_programs(4) > 1) else 0
    
    # Adjust for multi-dimensional grid if needed
    if tl.num_programs(3) > 1:
        out_w_idx = tl.program_id(3)
    if tl.num_programs(4) > 1:
        out_d_idx = tl.program_id(4)
    
    # Calculate starting input position
    in_w_start = out_w_idx * stride_w - pad_w
    in_h_start = out_h_idx * stride_h - pad_h
    in_d_start = out_d_idx * stride_d - pad_d
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for ic in range(in_channels):
        for kw in range(kernel_w):
            in_w = in_w_start + kw * dil_w
            if in_w < 0 or in_w >= input_w:
                continue
                
            for kh in range(kernel_h):
                in_h = in_h_start + kh * dil_h
                if in_h < 0 or in_h >= input_h:
                    continue
                    
                for kd in range(kernel_d):
                    in_d = in_d_start + kd * dil_d
                    if in_d < 0 or in_d >= input_d:
                        continue
                    
                    # Calculate input pointer offset
                    input_offset = (pid_batch * (in_channels * input_w * input_h * input_d) +
                                   ic * (input_w * input_h * input_d) +
                                   in_h * (input_w * input_d) +
                                   in_w * input_d +
                                   in_d)
                    
                    # Load input value
                    x_val = tl.load(x_ptr + input_offset)
                    
                    # Calculate weight pointer offset
                    weight_offset = (pid_out_ch * (in_channels * kernel_w * kernel_h * kernel_d) +
                                   ic * (kernel_w * kernel_h * kernel_d) +
                                   kw * (kernel_h * kernel_d) +
                                   kh * kernel_d +
                                   kd)
                    
                    # Load weight value
                    w_val = tl.load(w_ptr + weight_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + pid_out_ch)
        acc += bias_val
    
    # Store output
    output_offset = (pid_batch * (out_channels * output_w * output_h * output_d) +
                    pid_out_ch * (output_w * output_h * output_d) +
                    out_h_idx * (output_w * output_d) +
                    out_w_idx * output_d +
                    out_d_idx)
    
    tl.store(out_ptr + output_offset, acc.to(x_ptr.dtype.element_ty))


def triton_conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Performs 3D convolution using Triton kernel.
    
    Args:
        x: Input tensor [batch, in_channels, width, height, depth]
        weight: Weight tensor [out_channels, in_channels, kernel_w, kernel_h, kernel_d]
        bias: Optional bias tensor [out_channels]
        stride, padding, dilation, groups: Convolution parameters
    
    Returns:
        Output tensor
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, input_w, input_h, input_d = x.shape
    out_channels, _, kernel_w, kernel_h, kernel_d = weight.shape
    
    # Handle padding, stride, dilation
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    
    stride_w, stride_h, stride_d = stride
    pad_w, pad_h, pad_d = padding
    dil_w, dil_h, dil_d = dilation
    
    # Calculate output dimensions
    output_w = (input_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
    output_h = (input_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    output_d = (input_d + 2 * pad_d - dil_d * (kernel_d - 1) - 1) // stride_d + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, output_w, output_h, output_d, dtype=x.dtype, device=x.device)
    
    # Grid dimensions
    grid = (batch_size, out_channels, output_h, output_w, output_d)
    
    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        input_w, input_h, input_d,
        kernel_w, kernel_h, kernel_d,
        stride_w, stride_h, stride_d,
        pad_w, pad_h, pad_d,
        dil_w, dil_h, dil_d,
        output_w, output_h, output_d,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=1,
        BLOCK_SIZE_K=1
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model with custom Triton kernel for 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        """
        # Use the original conv3d weights and bias but replace the forward pass with Triton kernel
        return triton_conv3d(
            x, 
            self.conv3d.weight,
            self.conv3d.bias if self.conv3d.bias is not None else None,
            stride=self.conv3d.stride,
            padding=self.conv3d.padding,
            dilation=self.conv3d.dilation,
            groups=self.conv3d.groups
        )