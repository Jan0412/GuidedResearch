import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H, W, D)
    w_ptr,  # Weight tensor: (C_out, C_in, K_h, K_w, 1)
    b_ptr,  # Bias tensor: (C_out,)
    out_ptr,  # Output tensor: (B, C_out, H_out, W_out, D)
    B, C_in, H, W, D,
    C_out, K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w, x_stride_d,
    w_stride_c_out, w_stride_c_in, w_stride_kh, w_stride_kw, w_stride_kd,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w, out_stride_d,
    # Block sizes
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_KH: tl.constexpr, BLOCK_KW: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # Program IDs
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)
    depth_idx = tl.program_id(4)
    
    # Compute output position
    out_h_start = out_h * BLOCK_H
    out_w_start = out_w * BLOCK_W
    depth_start = depth_idx * BLOCK_D
    
    # Create offsets for output
    h_offsets = tl.arange(0, BLOCK_H)
    w_offsets = tl.arange(0, BLOCK_W)
    d_offsets = tl.arange(0, BLOCK_D)
    
    out_h_mask = (out_h_start + h_offsets) < (H - K_h + 2 * pad_h) // stride_h + 1
    out_w_mask = (out_w_start + w_offsets) < (W - K_w + 2 * pad_w) // stride_w + 1
    d_mask = depth_start + d_offsets < D
    
    # Output offsets
    out_h_grid, out_w_grid, d_grid = tl.meshgrid(out_h_mask, out_w_mask, d_mask)
    out_h_grid = out_h_grid * stride_h - pad_h + out_h_start
    out_w_grid = out_w_grid * stride_w - pad_w + out_w_start
    
    # Compute convolution
    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_D), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                # Compute input position
                h_pos = out_h_grid + kh * dil_h
                w_pos = out_w_grid + kw * dil_w
                
                # Check bounds
                h_in_mask = (h_pos >= 0) & (h_pos < H)
                w_in_mask = (w_pos >= 0) & (w_pos < W)
                in_mask = h_in_mask & w_in_mask
                
                # Load input
                x_offset = batch_idx * x_stride_b + c_in * x_stride_c + h_pos * x_stride_h + w_pos * x_stride_w + depth_start * x_stride_d
                x_val = tl.load(x_ptr + x_offset, mask=in_mask, other=0.0)
                
                # Load weight
                w_offset = out_c_idx * w_stride_c_out + c_in * w_stride_c_in + kh * w_stride_kh + kw * w_stride_kw + 0 * w_stride_kd
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_c_idx)
        acc += bias
    
    # Store result
    out_offset = (batch_idx * out_stride_b + out_c_idx * out_stride_c + 
                  (out_h_start + tl.arange(0, BLOCK_H)[:, None, None]) * out_stride_h + 
                  (tl.arange(0, BLOCK_W)[None, :, None]) * out_stride_w + 
                  (depth_start + tl.arange(0, BLOCK_D)[None, None, :]) * out_stride_d)
    
    out_mask = (out_h_mask[:, None, None] & out_w_mask[None, :, None] & d_mask[None, None, :])
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


class TritonConv3DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Extract dimensions
        B, C_in, H, W, D = x.shape
        C_out, _, K_h, K_w, _ = weight.shape
        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
        dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        # Calculate output dimensions
        H_out = (H + 2 * pad_h - K_h) // stride_h + 1
        W_out = (W + 2 * pad_w - K_w) // stride_w + 1
        
        # Allocate output
        out = torch.empty(B, C_out, H_out, W_out, D, device=x.device, dtype=x.dtype)
        
        # Grid dimensions
        grid = lambda meta: (
            B,
            C_out,
            triton.cdiv(H_out, meta['BLOCK_H']),
            triton.cdiv(W_out, meta['BLOCK_W']),
            triton.cdiv(D, meta['BLOCK_D'])
        )
        
        # Block sizes
        BLOCK_H = min(8, H_out)
        BLOCK_W = min(8, W_out)
        BLOCK_D = min(4, D)
        BLOCK_KH = K_h
        BLOCK_KW = K_w
        BLOCK_C = min(16, C_in)
        
        # Strides
        x_stride_b, x_stride_c, x_stride_h, x_stride_w, x_stride_d = x.stride()
        w_stride_c_out, w_stride_c_in, w_stride_kh, w_stride_kw, w_stride_kd = weight.stride()
        out_stride_b, out_stride_c, out_stride_h, out_stride_w, out_stride_d = out.stride()
        
        # Launch kernel
        conv3d_kernel[grid](
            x, weight, bias, out,
            B, C_in, H, W, D,
            C_out, K_h, K_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dil_h, dil_w,
            x_stride_b, x_stride_c, x_stride_h, x_stride_w, x_stride_d,
            w_stride_c_out, w_stride_c_in, w_stride_kh, w_stride_kw, w_stride_kd,
            out_stride_b, out_stride_c, out_stride_h, out_stride_w, out_stride_d,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_D=BLOCK_D,
            BLOCK_KH=BLOCK_KH, BLOCK_KW=BLOCK_KW, BLOCK_C=BLOCK_C
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_shape = x.shape
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass if needed, but for now we'll just use default
        raise NotImplementedError("Backward pass not implemented for TritonConv3DFunction")


class ModelNew(nn.Module):
    """
    Optimized version of the Model class using custom Triton kernels for 3D convolution.
    
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel (kernel_size x kernel_size).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
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
        
        # Initialize weights (same as nn.Conv3d)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming initialization similar to nn.Conv3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
        """
        import math
        
        # Use our custom Triton implementation
        return TritonConv3DFunction.apply(x, self.weight, self.bias, 
                                         self.stride, self.padding, self.dilation, self.groups)