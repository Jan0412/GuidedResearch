import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor: (B, C, H, W)
    w_ptr,  # Weight tensor: (C, 1, KH, KW)
    out_ptr,  # Output tensor: (B, C, H_out, W_out)
    B, C, H, W,
    KH, KW,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    H_out, W_out,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # Program ID for batch
    batch_id = tl.program_id(0)
    # Program ID for channel block
    c_block_id = tl.program_id(1)
    # Program ID for output height position
    h_out_id = tl.program_id(2)
    
    # Calculate output height position
    out_h = h_out_id
    
    # Calculate the starting input position for this output position
    in_h_start = out_h * stride_h - padding_h
    
    # Channel block
    c_start = c_block_id * BLOCK_SIZE_C
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    # Load input values for this position
    for c_idx in range(BLOCK_SIZE_C):
        if c_offsets[c_idx] < C:
            c = c_offsets[c_idx]
            # Process output width positions in a loop since we can't parallelize over W easily
            for w_out_id in range(W_out):
                out_w = w_out_id
                in_w_start = out_w * stride_w - padding_w
                
                # Compute the depthwise convolution
                acc = 0.0
                for kh in range(KH):
                    in_h = in_h_start + kh * dilation_h
                    if 0 <= in_h < H:
                        for kw in range(KW):
                            in_w = in_w_start + kw * dilation_w
                            if 0 <= in_w < W:
                                # Input index: (batch_id, c, in_h, in_w)
                                x_offset = batch_id * C * H * W + c * H * W + in_h * W + in_w
                                w_offset = c * KH * KW + kh * KW + kw
                                x_val = tl.load(x_ptr + x_offset)
                                w_val = tl.load(w_ptr + w_offset)
                                acc += x_val * w_val
                
                # Store result
                out_offset = batch_id * C * H_out * W_out + c * H_out * W_out + out_h * W_out + out_w
                tl.store(out_ptr + out_offset, acc)

class DepthwiseConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        
        # Extract dimensions
        B, C, H, W = x.shape
        KH, KW = weight.shape[2], weight.shape[3]
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation
        
        # Calculate output dimensions
        H_out = (H + 2 * padding_h - dilation_h * (KH - 1) - 1) // stride_h + 1
        W_out = (W + 2 * padding_w - dilation_w * (KW - 1) - 1) // stride_w + 1
        
        # Create output tensor
        out = torch.empty(B, C, H_out, W_out, dtype=x.dtype, device=x.device)
        
        # Set block sizes for grid
        BLOCK_SIZE_C = min(32, C)  # Tune this based on hardware
        BLOCK_SIZE_H = 1
        
        # Grid configuration
        grid = (B, (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C, H_out)
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, weight, out,
            B, C, H, W,
            KH, KW,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            H_out, W_out,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight)
        ctx.input_shape = (B, C, H, W)
        ctx.kernel_size = (KH, KW)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.output_shape = out.shape
        
        return out

def triton_depthwise_conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    return DepthwiseConv2dFunction.apply(x, weight, bias, stride, padding, dilation)

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.
    Optimized with Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        
        # Create parameters - for depthwise conv, groups=in_channels
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform depthwise convolution using Triton kernel
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias,
            stride=(self.stride_h, self.stride_w),
            padding=(self.padding_h, self.padding_w),
            dilation=(self.dilation_h, self.dilation_w)
        )