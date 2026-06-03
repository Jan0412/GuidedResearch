import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose1d_kernel(
    x_ptr,                # Input tensor (batch, in_channels, length)
    w_ptr,                # Weight tensor (in_channels, out_channels, kernel_size)
    b_ptr,                # Bias tensor (out_channels,) - can be None
    y_ptr,                # Output tensor (batch, out_channels, out_length)
    batch_size,           # Batch size
    in_channels,          # Number of input channels
    out_channels,         # Number of output channels
    input_length,         # Input length
    output_length,        # Output length
    kernel_size,          # Kernel size
    stride,               # Stride
    padding,              # Padding
    dilation,             # Dilation
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_L: tl.constexpr,  # Block size for sequence length
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_seq = tl.program_id(2)
    
    # Compute output position
    out_idx = pid_seq * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    out_mask = out_idx < output_length
    
    # Compute bias offset for this output channel
    bias_ptr = b_ptr + pid_out_c if b_ptr is not None else None
    
    # Accumulator for output
    acc = tl.zeros((BLOCK_SIZE_L,), dtype=tl.float32)
    
    # Loop over input channels
    for in_c in range(in_channels):
        # Compute input position for each output position
        # For transposed convolution: input_idx = (out_idx - (kernel_size-1)*dilation + padding) // stride
        # But we need to handle the full convolution effect
        
        # For each output position, compute which input positions contribute
        # out_idx = input_idx * stride + (kernel_pos - 1) * dilation - padding
        # => input_idx = (out_idx + padding - (kernel_pos - 1) * dilation) / stride
        
        # We'll iterate over kernel positions and compute the corresponding input indices
        for k in range(kernel_size):
            # Compute the input index that contributes to this output position
            # out_idx = input_idx * stride + k * dilation - padding
            # input_idx = (out_idx + padding - k * dilation) / stride
            input_idx = (out_idx + padding - k * dilation) // stride
            
            # Check if input_idx is valid
            input_mask = (input_idx >= 0) & (input_idx < input_length)
            
            # Check if the division was exact (no fractional input index)
            exact_div = ((out_idx + padding - k * dilation) % stride == 0)
            valid_mask = input_mask & exact_div
            
            # Load input values
            x_offsets = pid_batch * (in_channels * input_length) + in_c * input_length + input_idx
            x_val = tl.load(x_ptr + x_offsets, mask=valid_mask, other=0.0)
            
            # Load weight values
            w_offsets = in_c * (out_channels * kernel_size) + pid_out_c * kernel_size + k
            w_val = tl.load(w_ptr + w_offsets)
            
            # Accumulate
            acc += tl.where(valid_mask, x_val * w_val, 0.0)
    
    # Add bias if present
    if b_ptr is not None:
        bias_val = tl.load(bias_ptr)
        acc += bias_val
    
    # Store result
    y_offsets = pid_batch * (out_channels * output_length) + pid_out_c * output_length + out_idx
    tl.store(y_ptr + y_offsets, acc.to(tl.float32), mask=out_mask)


class TritonConvTranspose1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation):
        batch_size, in_channels, input_length = x.shape
        out_channels, _, kernel_size = weight.shape
        
        # Calculate output length: L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        # For our case with output_padding=0: L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
        output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
        
        # Allocate output tensor
        y = torch.empty(batch_size, out_channels, output_length, dtype=x.dtype, device=x.device)
        
        # Configure grid
        grid = lambda meta: (
            batch_size,
            triton.cdiv(out_channels, meta['BLOCK_SIZE_N']),
            triton.cdiv(output_length, meta['BLOCK_SIZE_L'])
        )
        
        # Launch kernel
        conv_transpose1d_kernel[grid](
            x, weight, bias, y,
            batch_size, in_channels, out_channels,
            input_length, output_length, kernel_size,
            stride, padding, dilation,
            BLOCK_SIZE_M=1,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_K=1,
            BLOCK_SIZE_L=128
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        
        # Compute gradients
        grad_x = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Gradient w.r.t. input - use transposed convolution with flipped weights
            grad_x = torch.nn.functional.conv1d(
                grad_output, weight.flip(-1), None,
                stride=dilation, padding=0, dilation=stride
            )
        
        if ctx.needs_input_grad[1]:
            # Gradient w.r.t. weight
            grad_weight = torch.zeros_like(weight)
            
            batch_size, in_channels, input_length = x.shape
            _, out_channels, output_length = grad_output.shape
            _, _, kernel_size = weight.shape
            
            # For each output position, accumulate gradients
            for b in range(batch_size):
                for oc in range(out_channels):
                    for ic in range(in_channels):
                        for k in range(kernel_size):
                            # Compute input index that contributes to this output
                            for o in range(output_length):
                                i = (o + padding - k * dilation) // stride
                                if ((o + padding - k * dilation) % stride == 0 and 
                                    0 <= i < input_length):
                                    grad_weight[ic, oc, k] += grad_output[b, oc, o] * x[b, ic, i]
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2))
        
        return grad_x, grad_weight, grad_bias, None, None, None


class ModelNew(nn.Module):
    """
    Optimized version of the transposed 1D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights with kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 1D convolution using Triton kernel.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        self.weight = self.weight.contiguous()
        
        # Call the Triton-based function
        return TritonConvTranspose1d.apply(x, self.weight, self.bias, 
                                          self.stride, self.padding, self.dilation)
    
    def _apply(self, fn):
        # Ensure parameters stay contiguous when moved between devices
        if self.weight is not None:
            self.weight.data = fn(self.weight.data)
        if self.bias is not None:
            self.bias.data = fn(self.bias.data)
        return super()._apply(fn)


import math