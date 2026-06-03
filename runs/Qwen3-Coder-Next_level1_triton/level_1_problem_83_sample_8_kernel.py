import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv1d_kernel(
    x_ptr,              # Input tensor pointer (B, C, H, W)
    w_ptr,              # Weight tensor pointer (C, 1, kernel_size)
    b_ptr,              # Bias tensor pointer (C,) or None
    out_ptr,            # Output tensor pointer (B, C, H, W_out)
    batch_size,         # B
    in_channels,        # C
    height,             # H
    width,              # W
    out_width,          # W_out
    kernel_size,        # K
    stride,             # S
    padding,            # P
    dilation,           # D
    BLOCK_SIZE_W: tl.constexpr,  # Block size for width dimension
    BLOCK_SIZE_C: tl.constexpr,  # Block size for channels
):
    # Program IDs
    pid_b = tl.program_id(0)  # batch
    pid_c = tl.program_id(1)  # channel
    pid_h = tl.program_id(2)  # height
    pid_w = tl.program_id(3)  # width block index
    
    # Compute the starting width index for this block
    w_start = pid_w * BLOCK_SIZE_W
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    w_mask = w_offsets < out_width
    
    # Compute the corresponding input width positions for each output position
    # For each output position w_out, the input positions are: 
    # w_in = w_out * stride - padding + k * dilation for k in [0, kernel_size)
    # We precompute the base offset for each output position
    base_w_in = w_offsets * stride - padding
    
    # Load the weights for this channel
    # Weight shape: (in_channels, 1, kernel_size)
    w_offsets_k = tl.arange(0, kernel_size)
    w_ptr_c = w_ptr + pid_c * kernel_size
    w_vals = tl.load(w_ptr_c + w_offsets_k, mask=w_offsets_k < kernel_size, other=0.0)
    
    # Load bias if present
    bias_val = 0.0
    if b_ptr is not None:
        bias_ptr = b_ptr + pid_c
        bias_val = tl.load(bias_ptr)
    
    # Compute the output for this block
    out_offsets = w_offsets
    out_mask = w_mask
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_W,), dtype=tl.float32)
    
    # Iterate over kernel positions
    for k in range(kernel_size):
        # Compute input width position for this kernel element
        w_in = base_w_in + k * dilation
        w_in_mask = (w_in >= 0) & (w_in < width)
        
        # Compute input pointer offset
        # Input shape: (batch_size, in_channels, height, width)
        # For fixed batch, channel, height: offset = w_in
        x_offset = w_in
        x_ptr_batch_channel_height = x_ptr + (pid_b * in_channels * height * width + 
                                               pid_c * height * width + 
                                               pid_h * width)
        
        # Load input values
        x_vals = tl.load(x_ptr_batch_channel_height + x_offset, mask=w_in_mask, other=0.0)
        
        # Accumulate: x_vals * w_vals[k]
        acc += x_vals * w_vals[k]
    
    # Add bias
    acc += bias_val
    
    # Store result
    tl.store(out_ptr + pid_b * in_channels * height * out_width + 
                    pid_c * height * out_width + 
                    pid_h * out_width + 
                    w_offsets, acc, mask=out_mask)

class DepthwiseConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, kernel_size, stride, padding, dilation):
        batch_size, in_channels, height, width = x.shape
        out_width = (width - kernel_size + 2 * padding) // stride + 1
        
        # Create output tensor
        out = torch.empty((batch_size, in_channels, height, out_width), 
                         dtype=x.dtype, device=x.device)
        
        # Determine grid dimensions
        # We'll use 4D grid: (batch, channel, height, width_block)
        BLOCK_SIZE_W = 32
        BLOCK_SIZE_C = 1  # For depthwise, process one channel at a time per block
        
        grid = (batch_size, in_channels, height, triton.cdiv(out_width, BLOCK_SIZE_W))
        
        # Launch kernel
        depthwise_conv1d_kernel[grid](
            x, weight, bias, out,
            batch_size, in_channels, height, width, out_width,
            kernel_size, stride, padding, dilation,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_C=BLOCK_SIZE_C
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a simplified implementation - full backward would require more complex kernels
        # For now, fall back to PyTorch's implementation for backward
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Use PyTorch's conv2d backward for gradient computation
            grad_input = torch.nn.grad.conv2d_input(x.shape, weight, grad_output, 
                                                   ctx.stride, ctx.padding, ctx.dilation, 
                                                   groups=x.size(1))
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(x, weight.shape, grad_output, 
                                                     ctx.stride, ctx.padding, ctx.dilation, 
                                                     groups=x.size(1))
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the weight and bias as in the original Conv2d
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight tensor with same shape as original Conv2d: (in_channels, 1, kernel_size, 1)
        # But we'll use only the kernel_size dimension along width
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get dimensions
        batch_size, in_channels, height, width = x.shape
        
        # Calculate output width
        out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        # Prepare output tensor
        out = torch.empty((batch_size, in_channels, height, out_width), 
                         dtype=x.dtype, device=x.device)
        
        # Define grid for 4D parallelization
        BLOCK_SIZE_W = 32
        grid = (batch_size, in_channels, height, triton.cdiv(out_width, BLOCK_SIZE_W))
        
        # Launch custom Triton kernel
        depthwise_conv1d_kernel[grid](
            x, self.weight, self.bias, out,
            batch_size, in_channels, height, width, out_width,
            self.kernel_size, self.stride, self.padding, self.dilation,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_C=1
        )
        
        return out