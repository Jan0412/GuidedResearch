import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,              # Input tensor pointer (N, C, H, W)
    w_ptr,              # Weight tensor pointer (C, 1, K, K)
    b_ptr,              # Bias tensor pointer (C,) - optional
    out_ptr,            # Output tensor pointer (N, C, H_out, W_out)
    n, c, h, w,         # Input dimensions
    h_out, w_out,       # Output dimensions
    kernel_size,        # Kernel size (square)
    stride,             # Stride
    padding,            # Padding
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    # Get the program IDs for batch, channel, and spatial position
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output spatial coordinates
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Calculate input spatial coordinates based on stride and padding
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    
    # Create masks for valid output indices
    mask_h = out_h < h_out
    mask_w = out_w < w_out
    mask_hw = mask_h[:, None] & mask_w[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over kernel height
    for kh in range(KERNEL_SIZE):
        # Calculate input height index
        in_h = in_h_start + kh
        
        # Mask for valid input height
        mask_h_k = (in_h >= 0) & (in_h < h)
        
        # Loop over kernel width
        for kw in range(KERNEL_SIZE):
            # Calculate input width index
            in_w = in_w_start + kw
            
            # Mask for valid input width
            mask_w_k = (in_w >= 0) & (in_w < w)
            mask_hw_k = mask_h_k[:, None] & mask_w_k[None, :] & mask_hw
            
            # Load input values (N, C, H, W format)
            # Calculate input pointer offset
            in_ptr = x_ptr + pid_n * (c * h * w) + pid_c * (h * w) + in_h[:, None] * w + in_w[None, :]
            
            # Load input with masking
            x_val = tl.load(in_ptr, mask=mask_hw_k, other=0.0)
            
            # Load weight value (C, 1, K, K format)
            w_val = tl.load(w_ptr + pid_c * kernel_size * kernel_size + kh * kernel_size + kw)
            
            # Accumulate
            acc += x_val * w_val
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_c)
        acc += bias
    
    # Store result
    out_ptr_offset = pid_n * (c * h_out * w_out) + pid_c * (h_out * w_out) + out_h[:, None] * w_out + out_w[None, :]
    tl.store(out_ptr + out_ptr_offset, acc.to(x_ptr.dtype.element_ty), mask=mask_hw)


class TritonDepthwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, stride=1, padding=0):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        
        # Get dimensions
        n, c, h, w = x.shape
        c_out, _, kernel_h, kernel_w = weight.shape
        
        # Calculate output dimensions
        h_out = (h + 2 * padding - kernel_h) // stride + 1
        w_out = (w + 2 * padding - kernel_w) // stride + 1
        
        # Create output tensor
        out = torch.empty((n, c_out, h_out, w_out), dtype=x.dtype, device=x.device)
        
        # Set up kernel launch parameters
        BLOCK_SIZE_H = 8
        BLOCK_SIZE_W = 8
        
        grid = (
            n,           # batch size
            c,           # channels
            (h_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
            (w_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
        )
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, weight, bias, out,
            n, c, h, w,
            h_out, w_out,
            kernel_h, stride, padding,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            KERNEL_SIZE=kernel_h,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_size = (n, c, h, w)
        ctx.output_size = (n, c_out, h_out, w_out)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass using PyTorch for simplicity
        # In production, you might want to implement custom backward kernels too
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        
        # Use PyTorch's native gradient computation
        grad_x = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_x = torch.nn.grad.conv2d_input(x.shape, weight, grad_output, 
                                               stride=stride, padding=padding)
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(x, weight.shape, grad_output,
                                                     stride=stride, padding=padding)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_x, grad_weight, grad_bias, None, None


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0):
    return TritonDepthwiseConv2d.apply(x, weight, bias, stride, padding)


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # For depthwise convolution, in_channels should equal out_channels and groups=in_channels
        # But we'll support arbitrary out_channels as well
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create weight and bias parameters
        # For depthwise: weight shape is (out_channels, in_channels // groups, kH, kW)
        # With groups=in_channels, this becomes (out_channels, 1, kH, kW)
        self.weight = nn.Parameter(torch.Tensor(out_channels, 1, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights using kaiming uniform for depthwise conv
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call our optimized Triton implementation
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)