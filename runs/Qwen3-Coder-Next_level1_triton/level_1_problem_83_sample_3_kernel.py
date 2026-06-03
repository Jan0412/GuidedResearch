import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv1d_kernel(
    x_ptr,  # Input tensor: (batch_size, in_channels, height, width)
    w_ptr,  # Weight tensor: (in_channels, kernel_size)
    b_ptr,  # Bias tensor: (in_channels,)
    out_ptr,  # Output tensor: (batch_size, in_channels, height, out_width)
    batch_size, in_channels, height, width, out_width, kernel_size, stride, padding, dilation,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    # Each program handles a contiguous block of (batch, channel, height) or width
    # We'll parallelize over batch, channel, height and process width in blocks
    
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_height = tl.program_id(2)
    
    # Compute output width index
    out_width_idx = tl.program_id(3)
    
    # Compute input position
    input_width = out_width_idx * stride + padding
    input_height = pid_height
    
    # Create ranges for width processing
    offsets_w = tl.arange(0, BLOCK_SIZE_W)
    out_width_offsets = out_width_idx * BLOCK_SIZE_W + offsets_w
    
    # Check if within bounds for output width
    mask_w = out_width_offsets < out_width
    
    # Load bias if present
    bias_val = 0.0
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + pid_channel)
    
    # Compute the convolution sum
    acc = tl.zeros((BLOCK_SIZE_W,), dtype=tl.float32) + bias_val
    
    # Loop over kernel elements
    for k in range(kernel_size):
        input_pos = input_width - k * dilation
        # Check if within valid input width range
        mask_valid = (input_pos >= 0) & (input_pos < width)
        
        # Compute input pointer offset for this position
        # Input layout: (batch_size, in_channels, height, width)
        offset = (pid_batch * in_channels * height * width + 
                 pid_channel * height * width + 
                 input_height * width + 
                 input_pos)
        
        # Load input value (if valid) and weight
        x_val = tl.load(x_ptr + offset, mask=mask_valid, other=0.0)
        w_val = tl.load(w_ptr + pid_channel * kernel_size + k)
        
        # Accumulate
        acc += x_val * w_val
    
    # Store output
    out_offset = (pid_batch * in_channels * height * out_width + 
                  pid_channel * height * out_width + 
                  input_height * out_width + 
                  out_width_offsets)
    
    tl.store(out_ptr + out_offset, acc, mask=mask_w)


class DepthwiseConv2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, kernel_size, stride, padding, dilation):
        batch_size, in_channels, height, width = x.shape
        out_width = (width - kernel_size + 2 * padding) // stride + 1
        
        # Prepare output tensor
        out = torch.empty((batch_size, in_channels, height, out_width), 
                         dtype=x.dtype, device=x.device)
        
        # Grid configuration for parallelization
        # We parallelize over batch, channel, height, and width blocks
        BLOCK_SIZE_H = 1
        BLOCK_SIZE_W = 32  # Process 32 output width positions per block
        
        grid = lambda meta: (
            batch_size,
            in_channels,
            height,
            triton.cdiv(out_width, meta['BLOCK_SIZE_W'])
        )
        
        # Launch kernel
        depthwise_conv1d_kernel[grid](
            x, weight, bias, out,
            batch_size, in_channels, height, width, out_width, kernel_size,
            stride, padding, dilation,
            BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
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
        # This is a simplified implementation - full backward pass would require more complex kernels
        # For now, we'll fall back to PyTorch for backward computation
        x, weight, bias = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        
        # Use PyTorch's built-in backward for simplicity
        # In production, you'd want custom backward kernels too
        return torch.nn.functional.conv2d(
            x, weight.unsqueeze(-1).unsqueeze(-1), None, 
            stride, padding, dilation, x.size(0)
        ).grad, None, None, None, None, None, None, None


def depthwise_conv2d_triton(x, weight, bias=None, stride=1, padding=0, dilation=1):
    # Reshape to match expected input format for our kernel
    # Our kernel assumes weight shape (in_channels, kernel_size)
    # PyTorch stores weight as (out_channels, in_channels_per_group, kernel_size, kernel_width)
    # For depthwise: in_channels_per_group=1, kernel_width=1, so we squeeze those dimensions
    
    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract parameters
    batch_size, in_channels, height, width = x.shape
    kernel_size = weight.shape[2]  # Since weight is (out_c, in_c, k_h, k_w) and k_w=1
    
    # Reshape weight for our kernel (in_channels, kernel_size)
    weight_reshaped = weight.view(in_channels, kernel_size)
    
    # Compute output dimensions
    out_width = (width - kernel_size + 2 * padding) // stride + 1
    
    # Call the autograd function
    return DepthwiseConv2DFunction.apply(x, weight_reshaped, bias, kernel_size, stride, padding, dilation)


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for depthwise convolution.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep the same structure as original
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), 
                               stride=stride, padding=padding, dilation=dilation, 
                               groups=in_channels, bias=bias)
        # Store parameters for reference
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias
        
        # Initialize weights from the original conv layer
        with torch.no_grad():
            # Copy weights and bias if they exist
            if self.conv2d.weight is not None:
                self.weight = nn.Parameter(self.conv2d.weight.data.clone().squeeze(-1))
            if self.conv2d.bias is not None:
                self.bias = nn.Parameter(self.conv2d.bias.data.clone())
            else:
                self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 2D convolution using Triton kernel.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Call the optimized Triton implementation
        return depthwise_conv2d_triton(
            x, 
            self.weight, 
            self.bias if self.has_bias else None,
            self.stride, 
            self.padding, 
            self.dilation
        )