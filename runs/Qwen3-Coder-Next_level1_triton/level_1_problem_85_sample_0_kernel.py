import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer (B, C, H, W)
    w_ptr,  # Weight tensor pointer (C, 1, Kh, Kw)
    out_ptr,  # Output tensor pointer (B, C, H_out, W_out)
    batch_size,  # B
    in_channels,  # C
    in_h,  # Input height
    in_w,  # Input width
    out_h,  # Output height
    out_w,  # Output width
    kh,  # Kernel height
    kw,  # Kernel width
    stride_h,  # Stride height
    stride_w,  # Stride width
    padding_h,  # Padding height
    padding_w,  # Padding width
    dilation_h,  # Dilation height
    dilation_w,  # Dilation width
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get the program IDs
    bc = tl.program_id(0)  # Batch index
    c = tl.program_id(1)   # Channel index
    bh = tl.program_id(2)  # Block in height dimension
    bw = tl.program_id(3)  # Block in width dimension

    # Compute the output spatial position
    out_h_start = bh * BLOCK_SIZE_H
    out_w_start = bw * BLOCK_SIZE_W

    # Create ranges for output positions
    oh_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    ow_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Create mask for valid output positions
    oh_mask = oh_offsets < out_h
    ow_mask = ow_offsets < out_w
    output_mask = oh_mask[:, None] & ow_mask[None, :]
    
    # Initialize accumulator
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over kernel height
    for kh_idx in range(kh):
        # Compute input height position with dilation
        in_h_pos = (out_h_start * stride_h + kh_idx * dilation_h - padding_h)
        
        # Check if this input row is valid
        valid_h = (in_h_pos >= 0) & (in_h_pos < in_h)
        
        if valid_h:
            # Iterate over kernel width
            for kw_idx in range(kw):
                # Compute input width position with dilation
                in_w_pos = (out_w_start * stride_w + kw_idx * dilation_w - padding_w)
                
                # Check if this input column is valid
                valid_w = (in_w_pos >= 0) & (in_w_pos < in_w)
                
                if valid_w:
                    # Load input values
                    in_h_offset = in_h_pos
                    in_w_offset = in_w_pos
                    input_offsets = (
                        bc * (in_channels * in_h * in_w) +
                        c * (in_h * in_w) +
                        in_h_offset * in_w +
                        in_w_offset
                    )
                    
                    # Load input value (scalar for this position)
                    input_val = tl.load(x_ptr + input_offsets)
                    
                    # Load weight value
                    weight_offsets = (
                        c * (kh * kw) +
                        kh_idx * kw +
                        kw_idx
                    )
                    weight_val = tl.load(w_ptr + weight_offsets)
                    
                    # Accumulate the product
                    output += input_val * weight_val
    
    # Store the result
    output_offsets = (
        bc * (out_channels * out_h * out_w) +
        c * (out_h * out_w) +
        oh_offsets[:, None] * out_w +
        ow_offsets[None, :]
    )
    
    # Note: out_channels == in_channels for depthwise convolution
    tl.store(out_ptr + output_offsets, output, mask=output_mask)


class TritonDepthwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, stride_h=1, stride_w=1, padding_h=0, padding_w=0, dilation_h=1, dilation_w=1):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        
        # Extract dimensions
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels, _, kh, kw = weight.shape
        
        # Calculate output dimensions
        out_h = (in_h + 2 * padding_h - dilation_h * (kh - 1) - 1) // stride_h + 1
        out_w = (in_w + 2 * padding_w - dilation_w * (kw - 1) - 1) // stride_w + 1
        
        # Create output tensor
        out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
        
        # Set block sizes (tunable parameters)
        BLOCK_SIZE_C = 1
        BLOCK_SIZE_H = 8
        BLOCK_SIZE_W = 8
        
        # Calculate grid dimensions
        grid = (
            batch_size,  # batch
            in_channels,  # channels
            (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,  # height blocks
            (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W   # width blocks
        )
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, weight, out,
            batch_size, in_channels, in_h, in_w, out_h, out_w,
            kh, kw, stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
        )
        
        # Save for backward pass (though we won't implement backward for simplicity)
        ctx.save_for_backward(x, weight)
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.padding_h = padding_h
        ctx.padding_w = padding_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.in_h = in_h
        ctx.in_w = in_w
        
        # Add bias if provided
        if bias is not None:
            # Bias shape is (out_channels,), need to broadcast to (batch, out_channels, out_h, out_w)
            out = out + bias.view(1, -1, 1, 1)
            
        return out


def triton_depthwise_conv2d(x, weight, bias=None, stride_h=1, stride_w=1, padding_h=0, padding_w=0, dilation_h=1, dilation_w=1):
    return TritonDepthwiseConv2d.apply(x, weight, bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for reference
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
        self.groups = groups
        
        # Create weight and bias parameters
        # For depthwise conv: weight shape is (in_channels, 1, kh, kw)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))  # Depthwise has bias per channel
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = kernel_size_h * kernel_size_w
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use our custom Triton implementation for depthwise convolution
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias,
            self.stride_h, 
            self.stride_w, 
            self.padding_h, 
            self.padding_w, 
            self.dilation_h, 
            self.dilation_w
        )

# Import math for initialization
import math