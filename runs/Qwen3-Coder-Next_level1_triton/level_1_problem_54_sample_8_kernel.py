import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer (batch, in_channels, depth, height, width)
    w_ptr,  # Weight tensor pointer (out_channels, in_channels, k_d, k_h, k_w)
    b_ptr,  # Bias tensor pointer (out_channels,)
    out_ptr,  # Output tensor pointer
    batch_size, in_channels, out_channels,
    depth, height, width,
    k_d, k_h, k_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    out_d, out_h, out_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for spatial elements
    BLOCK_SIZE_K: tl.constexpr,  # Block size for reduction (in_channels * k_d * k_h * k_w)
):
    # Output tensor dimensions: (batch, out_channels, out_d, out_h, out_w)
    # Each block processes multiple output elements
    
    # Get program IDs
    pid_m = tl.program_id(0)  # For output channels
    pid_n = tl.program_id(1)  # For spatial position
    
    # Calculate which batch and spatial position this block handles
    # Flatten spatial dimensions for easier indexing
    total_spatial = out_d * out_h * out_w
    batch_idx = pid_n // total_spatial
    spatial_idx = pid_n % total_spatial
    
    # Convert flattened spatial index back to 3D coordinates
    out_d_idx = spatial_idx // (out_h * out_w)
    out_h_idx = (spatial_idx % (out_h * out_w)) // out_w
    out_w_idx = spatial_idx % out_w
    
    # Calculate input spatial coordinates (accounting for stride, padding, dilation)
    in_d_idx = out_d_idx * stride_d - pad_d + out_d_idx * (dil_d - 1)
    in_h_idx = out_h_idx * stride_h - pad_h + out_h_idx * (dil_h - 1)
    in_w_idx = out_w_idx * stride_w - pad_w + out_w_idx * (dil_w - 1)
    
    # Create offsets for output channels
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < out_channels
    
    # Create offsets for input channels and kernel elements (reduction dimension)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Iterate over reduction dimension (in_channels, k_d, k_h, k_w)
    # We'll process in_channels in chunks, then kernel dimensions
    for ic in range(in_channels):
        # For each input channel, iterate over kernel dimensions
        for kd in range(k_d):
            for kh in range(k_h):
                for kw in range(k_w):
                    # Calculate input position
                    in_d = in_d_idx + kd * dil_d
                    in_h = in_h_idx + kh * dil_h
                    in_w = in_w_idx + kw * dil_w
                    
                    # Check bounds
                    valid = (in_d >= 0) & (in_d < depth) & (in_h >= 0) & (in_h < height) & (in_w >= 0) & (in_w < width)
                    
                    if tl.any(valid):
                        # Calculate input pointer offset
                        input_offset = (batch_idx * in_channels * depth * height * width +
                                       ic * depth * height * width +
                                       in_d * height * width +
                                       in_h * width +
                                       in_w)
                        
                        # Calculate weight pointer offset
                        weight_offset = (m_offsets[:, None] * in_channels * k_d * k_h * k_w +
                                        ic * k_d * k_h * k_w +
                                        kd * k_h * k_w +
                                        kh * k_w +
                                        kw)
                        
                        # Load input value
                        x_val = tl.load(x_ptr + input_offset, mask=valid, other=0.0)
                        
                        # Load weight values
                        w_vals = tl.load(w_ptr + weight_offset, mask=m_mask[:, None], other=0.0)
                        
                        # Accumulate
                        accumulator += x_val * tl.sum(w_vals, axis=1)
    
    # Add bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + m_offsets, mask=m_mask, other=0.0)
        accumulator += bias
    
    # Store output
    out_offset = (batch_idx * out_channels * out_d * out_h * out_w +
                 m_offsets * out_d * out_h * out_w +
                 out_d_idx * out_h * out_w +
                 out_h_idx * out_w +
                 out_w_idx)
    
    tl.store(out_ptr + out_offset, accumulator.to(x_ptr.dtype.element_ty), mask=m_mask)


class TritonConv3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups):
        # Extract parameters
        batch_size, in_channels, depth, height, width = x.shape
        out_channels, _, k_d, k_h, k_w = weight.shape
        
        # Calculate output dimensions
        stride_d, stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride, stride)
        pad_d, pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding, padding)
        dil_d, dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        
        out_d = (depth + 2 * pad_d - dil_d * (k_d - 1) - 1) // stride_d + 1
        out_h = (height + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h + 1
        out_w = (width + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w + 1
        
        # Prepare output tensor
        out = torch.empty(batch_size, out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
        
        # Define kernel launch parameters
        BLOCK_SIZE_M = 8  # Output channels per block
        BLOCK_SIZE_N = 64  # Spatial elements per block
        BLOCK_SIZE_K = 32  # Reduction size
        
        # Calculate grid size
        grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = batch_size * ((out_d * out_h * out_w + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
        grid = (grid_m, grid_n)
        
        # Launch kernel
        conv3d_kernel[grid](
            x, weight, bias, out,
            batch_size, in_channels, out_channels,
            depth, height, width,
            k_d, k_h, k_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dil_d, dil_h, dil_w,
            out_d, out_h, out_w,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.input_shape = (batch_size, in_channels, depth, height, width)
        ctx.weight_shape = weight.shape
        ctx.output_shape = out.shape
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch's backward implementation
        x, weight, bias = ctx.saved_tensors
        
        # Compute gradients using PyTorch (simpler than implementing full backward in Triton)
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = F.grad3d_conv_forward(
                grad_output.contiguous(), x, weight,
                ctx.padding, ctx.stride, ctx.dilation, ctx.groups
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = F.grad3d_conv_weight(
                weight.shape, grad_output.contiguous(), x,
                ctx.padding, ctx.stride, ctx.dilation, ctx.groups
            )
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3, 4])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None


def triton_conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return TritonConv3dFunction.apply(x, weight, bias, stride, padding, dilation, groups)


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using custom Triton kernel.
        """
        return triton_conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)