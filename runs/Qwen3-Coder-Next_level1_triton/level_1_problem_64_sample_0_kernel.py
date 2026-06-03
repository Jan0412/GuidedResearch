import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, L_in)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, L_out)
    batch_size, in_channels, out_channels, kernel_size,
    stride, padding, output_padding,
    L_in, L_out,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_BLOCK: tl.constexpr,
):
    # Program IDs for batch, output channel, and output position
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_l_out = tl.program_id(2)
    
    # Calculate the input position that this output position depends on
    # For transposed convolution: out[l_out] += sum_{k} w[k] * x[(l_out - k) * stride + padding]
    # But we need to handle the inverse relationship: which input positions contribute to this output
    
    # Output position
    l_out = pid_l_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # If bias is provided, add it
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Iterate over input channels
    for c_in in range(in_channels):
        # Iterate over kernel positions
        for k in range(kernel_size):
            # Calculate corresponding input position
            # l_out = stride * l_in + (k - padding) + output_padding_offset
            # So l_in = (l_out - (k - padding + output_padding_offset)) / stride
            # Simplified: l_in = (l_out - k + padding) / stride
            
            l_in = (l_out - k + padding) // stride
            
            # Check bounds
            if l_in >= 0 and l_in < L_in:
                # Calculate the exact offset for input
                x_offset = pid_b * (in_channels * L_in) + c_in * L_in + l_in
                w_offset = c_in * (out_channels * kernel_size) + pid_c_out * kernel_size + k
                
                # Load input and weight
                x_val = tl.load(x_ptr + x_offset)
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc += x_val * w_val
    
    # Store result
    out_offset = pid_b * (out_channels * L_out) + pid_c_out * L_out + l_out
    tl.store(out_ptr + out_offset, acc.to(tl.float32))


def triton_conv_transpose1d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of 1D transposed convolution.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, L_in)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Bias tensor of shape (out_channels,) or None
        stride, padding, output_padding, groups: convolution parameters
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, L_in = x.shape
    _, out_channels, kernel_size = weight.shape
    
    # Calculate output length
    # L_out = (L_in - 1) * stride - 2 * padding + output_padding + kernel_size
    L_out = (L_in - 1) * stride - 2 * padding + output_padding + kernel_size
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, L_out, dtype=x.dtype, device=x.device)
    
    # Set block sizes
    BLOCK_SIZE = 128
    KERNEL_BLOCK = 32  # For kernel dimension
    
    # Grid: (batch_size, out_channels, L_out)
    grid = (batch_size, out_channels, L_out)
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, kernel_size,
        stride, padding, output_padding,
        L_in, L_out,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_BLOCK=KERNEL_BLOCK,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create weight and bias parameters (same as nn.ConvTranspose1d)
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        # Kaiming uniform initialization similar to PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is correct shape
        assert x.dim() == 3, "Input must be 3D: (batch, channels, length)"
        
        # Call our custom Triton kernel
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding,
            groups=self.groups
        )