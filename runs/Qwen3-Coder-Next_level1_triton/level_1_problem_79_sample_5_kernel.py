import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,  # Input: (batch, in_channels, L_in)
    w_ptr,  # Weight: (in_channels, out_channels, kernel_size)
    b_ptr,  # Bias: (out_channels,) or None
    out_ptr,  # Output: (batch, out_channels, L_out)
    batch_size, in_channels, out_channels, kernel_size,
    L_in, L_out,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,  # Block size for reduction dimension
):
    # Program ID maps to (batch, out_channel, position_in_output)
    pid_batch = tl.program_id(0)
    pid_out_channel = tl.program_id(1)
    pid_pos = tl.program_id(2)
    
    # Calculate output position
    out_pos = pid_pos
    
    # Calculate corresponding input positions using transposed convolution formula
    # For transposed conv: out_pos = in_pos * stride + (k - (kernel_size - 1) * dilation - 1) + padding
    # Rearranging: in_pos = (out_pos - (k - (kernel_size - 1) * dilation - 1) - padding) / stride
    
    # We'll iterate over kernel positions k in [0, kernel_size)
    # and input channel positions c_in in [0, in_channels)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(in_channels):
        # Loop over kernel positions
        for k in range(kernel_size):
            # Calculate corresponding input position
            # out_pos = in_pos * stride + k * dilation - padding
            # => in_pos = (out_pos - k * dilation + padding) / stride
            # Only valid if (out_pos - k * dilation + padding) is divisible by stride and in range
            
            offset = out_pos - k * dilation + padding
            if offset % stride == 0:
                in_pos = offset // stride
                if 0 <= in_pos < L_in:
                    # Load input value
                    x_offset = pid_batch * (in_channels * L_in) + c_in * L_in + in_pos
                    x_val = tl.load(x_ptr + x_offset)
                    
                    # Load weight value
                    w_offset = c_in * (out_channels * kernel_size) + pid_out_channel * kernel_size + k
                    w_val = tl.load(w_ptr + w_offset)
                    
                    # Accumulate
                    acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias_offset = pid_out_channel
        bias_val = tl.load(b_ptr + bias_offset)
        acc += bias_val
    
    # Store result
    out_offset = pid_batch * (out_channels * L_out) + pid_out_channel * L_out + out_pos
    tl.store(out_ptr + out_offset, acc)


class TritonConvTranspose1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        batch_size, in_channels, L_in = x.shape
        _, out_channels, kernel_size = weight.shape
        
        # Calculate output length: L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1
        # For standard transposed conv with out_padding=0: L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
        L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
        
        # Create output tensor
        out = torch.empty(batch_size, out_channels, L_out, dtype=x.dtype, device=x.device)
        
        # Define grid dimensions
        # Grid: (batch_size, out_channels, L_out)
        grid = (batch_size, out_channels, L_out)
        
        # Block size for output positions
        BLOCK_SIZE = 1
        BLOCK_K = 128  # Not used directly in this implementation but kept for flexibility
        
        # Launch kernel
        conv_transpose1d_kernel[grid](
            x, weight, bias, out,
            batch_size, in_channels, out_channels, kernel_size,
            L_in, L_out,
            stride, padding, dilation,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_K=BLOCK_K
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.L_in = L_in
        ctx.L_out = L_out
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass using PyTorch for simplicity (or could be optimized)
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        
        # Use PyTorch's built-in transposed conv backward
        # This is sufficient for training functionality
        grad_x = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Compute gradient w.r.t. input
            grad_x = torch.nn.grad.conv1d_input(x.shape, weight, grad_output, 
                                                stride=stride, padding=padding, 
                                                dilation=dilation, groups=1)
        
        if ctx.needs_input_grad[1]:
            # Compute gradient w.r.t. weight
            grad_weight = torch.nn.grad.conv1d_weight(x, weight.shape, grad_output,
                                                     stride=stride, padding=padding,
                                                     dilation=dilation, groups=1)
        
        if bias is not None and ctx.needs_input_grad[2]:
            # Compute gradient w.r.t. bias
            grad_bias = grad_output.sum(dim=[0, 2])
        
        return grad_x, grad_weight, grad_bias, None, None, None


def triton_conv_transpose1d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    return TritonConvTranspose1d.apply(x, weight, bias, stride, padding, dilation)


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 1D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming initialization for transposed conv
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is on the same device as our parameters
        x = x.to(self.weight.device)
        
        # Use our optimized Triton implementation
        return triton_conv_transpose1d(x, self.weight, self.bias, 
                                      self.stride, self.padding, self.dilation)