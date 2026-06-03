import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,              # Input tensor: (B, C_in, L_in)
    w_ptr,              # Weight tensor: (C_in, C_out, K)
    b_ptr,              # Bias tensor: (C_out,) or None
    out_ptr,            # Output tensor: (B, C_out, L_out)
    B: tl.constexpr,    # Batch size
    C_in: tl.constexpr, # Input channels
    C_out: tl.constexpr, # Output channels
    L_in: tl.constexpr, # Input length
    L_out: tl.constexpr, # Output length
    K: tl.constexpr,    # Kernel size
    stride: tl.constexpr, # Stride
    padding: tl.constexpr, # Padding
    output_padding: tl.constexpr, # Output padding
    BLOCK_SIZE_L: tl.constexpr,  # Block size for length dimension
    BLOCK_SIZE_C: tl.constexpr,  # Block size for channel dimension
):
    # Program IDs: 
    # pid_b: batch index
    # pid_c_out: output channel index
    # pid_l: output position index
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    # Calculate the output position
    out_pos = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    
    # Check bounds for output length
    mask_l = out_pos < L_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_L,), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in in range(C_in):
        # Calculate input position: out_pos corresponds to multiple input positions due to stride
        # For transposed conv: in_pos = (out_pos - padding + output_padding) // stride
        # But we need to handle the fractional positions properly
        in_pos_base = (out_pos - padding) // stride
        
        # For each kernel position
        for k in range(K):
            # Calculate input position with kernel offset
            in_pos = in_pos_base + k
            
            # Check bounds for input position
            mask_in = (in_pos >= 0) & (in_pos < L_in)
            
            # Calculate weight index: w[c_in, pid_c_out, k]
            w_idx = c_in * (C_out * K) + pid_c_out * K + k
            weight = tl.load(w_ptr + w_idx)
            
            # Calculate input index: x[pid_b, c_in, in_pos]
            x_idx = pid_b * (C_in * L_in) + c_in * L_in + in_pos
            x_val = tl.load(x_ptr + x_idx, mask=mask_in, other=0.0)
            
            # Accumulate: output += input * weight
            acc += tl.where(mask_in, x_val * weight, 0.0)
    
    # Apply bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c_out)
        acc += bias
    
    # Store result
    out_idx = pid_b * (C_out * L_out) + pid_c_out * L_out + out_pos
    tl.store(out_ptr + out_idx, acc.to(x_ptr.dtype.element_ty), mask=mask_l)


def triton_conv_transpose1d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose1d.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, length)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride for the transpose convolution
        padding: Padding applied to input
        output_padding: Additional size added to output
        groups: Number of groups (only groups=1 is supported in this kernel)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "Only groups=1 is supported in this kernel implementation."
    
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, L_in = x.shape
    C_out, _, K = weight.shape
    
    # Calculate output length: L_out = (L_in - 1) * stride - 2 * padding + output_padding + K
    L_out = (L_in - 1) * stride - 2 * padding + output_padding + K
    
    # Prepare output tensor
    out = torch.empty((B, C_out, L_out), dtype=x.dtype, device=x.device)
    
    # Define block sizes for parallelization
    BLOCK_SIZE_L = 256  # Block size for output length dimension
    BLOCK_SIZE_C = 1    # We parallelize over output channels directly
    
    # Grid dimensions: (batch_size, out_channels, ceil(L_out / BLOCK_SIZE_L))
    grid = (B, C_out, triton.cdiv(L_out, BLOCK_SIZE_L))
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        B=B, C_in=C_in, C_out=C_out,
        L_in=L_in, L_out=L_out, K=K,
        stride=stride, padding=padding, output_padding=output_padding,
        BLOCK_SIZE_L=BLOCK_SIZE_L, BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters and initialize weights
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weight and bias parameters
        # Note: nn.ConvTranspose1d uses shape (in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, 
            output_padding=self.output_padding, groups=self.groups
        )