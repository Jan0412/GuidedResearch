import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,            # Input tensor pointer (B, C, H, W)
    w_ptr,            # Weight tensor pointer (C, 1, K, K)
    b_ptr,            # Bias tensor pointer (C,) - optional
    y_ptr,            # Output tensor pointer (B, C, H_out, W_out)
    batch_size,       # Batch size
    channels,         # Number of channels (in_channels = out_channels for depthwise)
    height_in,        # Input height
    width_in,         # Input width
    height_out,       # Output height
    width_out,        # Output width
    kernel_size,      # Kernel size (square)
    stride,           # Stride
    padding,          # Padding
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Compute output position
    h_out = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out = tl.arange(0, BLOCK_W)
    
    # Create mask for valid output positions
    h_mask = h_out < height_out
    w_mask = w_out < width_out
    mask = h_mask[:, None] & w_mask[None, :]
    
    # Compute corresponding input positions
    h_in = h_out * stride - padding
    w_in = w_out * stride - padding
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_size):
        h_in_k = h_in + kh
        h_in_k_mask = (h_in_k >= 0) & (h_in_k < height_in)
        h_in_k = h_in_k * width_in  # For 1D indexing
        
        for kw in range(kernel_size):
            w_in_k = w_out + kw
            w_in_k_mask = (w_in_k >= 0) & (w_in_k < width_in)
            
            # Load input values for current kernel position
            h_idx = h_in_k[:, None] + w_in_k[None, :]
            h_idx = h_idx * channels + pid_c  # Add channel offset
            x_offset = pid_b * (height_in * width_in * channels) + h_idx
            
            # Load input
            x_val = tl.load(x_ptr + x_offset, mask=mask & h_in_k_mask[:, None] & w_in_k_mask[None, :], other=0.0)
            
            # Load weight
            w_offset = pid_c * (kernel_size * kernel_size) + kh * kernel_size + kw
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        b_val = tl.load(b_ptr + pid_c)
        acc += b_val
    
    # Store result
    y_offset = pid_b * (height_out * width_out * channels) + \
               (h_out[:, None] * width_out + w_out[None, :]) * channels + pid_c
    tl.store(y_ptr + y_offset, acc.to(x_ptr.dtype.element_ty), mask=mask)


class TritonDepthwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, stride=1, padding=0):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Extract dimensions
        batch_size, channels, height_in, width_in = x.shape
        _, _, kernel_size, _ = weight.shape
        
        # Compute output dimensions
        height_out = (height_in + 2 * padding - kernel_size) // stride + 1
        width_out = (width_in + 2 * padding - kernel_size) // stride + 1
        
        # Allocate output tensor
        y = torch.empty((batch_size, channels, height_out, width_out), 
                       dtype=x.dtype, device=x.device)
        
        # Configure kernel launch parameters
        # We use a 3D grid: (batch, channel, height_blocks)
        BLOCK_H = 8
        BLOCK_W = 32
        BLOCK_K = 1  # Not used in kernel but kept for interface
        
        grid = (batch_size, channels, (height_out + BLOCK_H - 1) // BLOCK_H)
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, weight, bias, y,
            batch_size, channels, height_in, width_in,
            height_out, width_out, kernel_size, stride, padding,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_K=BLOCK_K
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.kernel_size = kernel_size
        ctx.height_out = height_out
        ctx.width_out = width_out
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # Not implementing backward for simplicity - PyTorch will handle it via autograd
        # For production use, you'd implement backward pass for full functionality
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Compute grad_input using transposed convolution logic
            grad_input = torch.nn.grad.conv2d_input(
                x.shape, weight, grad_output, 
                stride=ctx.stride, padding=ctx.padding
            )
        
        if ctx.needs_input_grad[1]:
            # Compute grad_weight
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output, 
                stride=ctx.stride, padding=ctx.padding
            )
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # For depthwise convolution, in_channels == out_channels == groups
        # Note: We'll use the same number of channels for both in and out
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Register weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming initialization for depthwise convolution
        nn.init.kaiming_uniform_(self.weight, a=0)  # mode='fan_in' for depthwise
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is the right shape and on the right device
        x = x.contiguous()
        
        # Use our custom Triton implementation
        return TritonDepthwiseConv2d.apply(x, self.weight, self.bias, 
                                          self.stride, self.padding)