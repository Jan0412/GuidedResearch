import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,                # Input tensor pointer (B, C, H, W)
    w_ptr,                # Weight tensor pointer (C, 1, K, K)
    b_ptr,                # Bias tensor pointer (C,) - optional
    out_ptr,              # Output tensor pointer (B, C, H_out, W_out)
    batch_size,           # B
    in_channels,          # C
    in_height,            # H
    in_width,             # W
    out_height,           # H_out
    out_width,            # W_out
    kernel_size,          # K
    stride,               # stride
    padding,              # padding
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the batch and channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate output spatial position
    out_h_start = tl.program_id(2) * BLOCK_SIZE_H
    out_w_start = tl.program_id(3) * BLOCK_SIZE_W
    
    # Compute input starting position based on stride and padding
    in_h_start = out_h_start * stride - padding
    in_w_start = out_w_start * stride - padding
    
    # Accumulator for the convolution
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input position
            in_h = in_h_start + kh
            in_w = in_w_start + kw
            
            # Check bounds for input height
            h_valid = (in_h >= 0) & (in_h < in_height)
            # Check bounds for input width
            w_valid = (in_w >= 0) & (in_w < in_width)
            
            # Compute offset for input tensor
            # Layout: B, C, H, W
            offset = (batch_idx * in_channels * in_height * in_width +
                     channel_idx * in_height * in_width +
                     in_h * in_width + in_w)
            
            # Load input value with bounds checking
            x_val = tl.load(x_ptr + offset, 
                          mask=(h_valid & w_valid), 
                          other=0.0)
            
            # Compute weight offset
            # Weight layout: C, 1, K, K
            w_offset = (channel_idx * kernel_size * kernel_size +
                       kh * kernel_size + kw)
            
            # Load weight
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        bias_offset = channel_idx
        bias_val = tl.load(b_ptr + bias_offset)
        acc += bias_val
    
    # Store results
    # Output layout: B, C, H_out, W_out
    for i in range(BLOCK_SIZE_H):
        out_h = out_h_start + i
        if out_h < out_height:
            for j in range(BLOCK_SIZE_W):
                out_w = out_w_start + j
                if out_w < out_width:
                    offset = (batch_idx * in_channels * out_height * out_width +
                             channel_idx * out_height * out_width +
                             out_h * out_width + out_w)
                    tl.store(out_ptr + offset, acc[i, j])


class TritonDepthwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, kernel_size, stride, padding):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        batch_size, in_channels, in_height, in_width = x.shape
        out_channels = weight.shape[0]  # For depthwise, out_channels == in_channels
        kernel_h, kernel_w = kernel_size, kernel_size
        
        # Calculate output dimensions
        out_height = (in_height + 2 * padding - kernel_h) // stride + 1
        out_width = (in_width + 2 * padding - kernel_w) // stride + 1
        
        # Create output tensor
        out = torch.empty(batch_size, out_channels, out_height, out_width, 
                         dtype=x.dtype, device=x.device)
        
        # Configure grid dimensions
        # Grid: (batch_size, in_channels, out_height_blocks, out_width_blocks)
        BLOCK_SIZE_H = 4
        BLOCK_SIZE_W = 32  # Optimized for width=512
        BLOCK_SIZE_K = 1
        
        # Calculate number of blocks needed
        grid_h = (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
        grid_w = (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
        
        grid = (batch_size, in_channels, grid_h, grid_w)
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, weight, bias, out,
            batch_size, in_channels, in_height, in_width,
            out_height, out_width,
            kernel_h, stride, padding,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.input_size = (in_height, in_width)
        ctx.output_size = (out_height, out_width)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch's autograd
        # A full implementation would require implementing backward pass in Triton
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Use PyTorch's native conv2d for backward
            grad_input = torch.nn.functional.conv_transpose2d(
                grad_output, weight, None, ctx.stride, ctx.padding,
                dilation=1, groups=x.shape[1], output_padding=0
            )
        
        if ctx.needs_input_grad[1]:
            # Compute gradient for weights using native conv2d
            grad_weight = torch.empty_like(weight)
            for i in range(weight.shape[0]):
                # For each channel, compute gradient
                x_i = x[:, i:i+1, :, :]  # (B, 1, H, W)
                grad_out_i = grad_output[:, i:i+1, :, :]  # (B, 1, H_out, W_out)
                
                # Use unfold to get patches
                patches = torch.nn.functional.unfold(
                    x_i, kernel_size=ctx.kernel_size, 
                    padding=ctx.padding, stride=ctx.stride
                )  # (B, K*K, L)
                
                grad_out_i_flat = grad_out_i.view(grad_out_i.shape[0], grad_out_i.shape[1], -1)  # (B, 1, L)
                
                # Compute gradient
                grad_weight_i = torch.bmm(grad_out_i_flat, patches.transpose(1, 2)).mean(dim=0)
                grad_weight[i, 0, :, :] = grad_weight_i.view(ctx.kernel_size, ctx.kernel_size)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using custom Triton kernel.
    
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # For depthwise convolution, groups=in_channels, so out_channels should equal in_channels
        # But we'll handle the general case for compatibility
        if out_channels != in_channels:
            raise ValueError("For depthwise convolution, out_channels must equal in_channels")
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming initialization for depthwise convolution
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonDepthwiseConv2d.apply(x, self.weight, self.bias, 
                                          self.kernel_size, self.stride, self.padding)