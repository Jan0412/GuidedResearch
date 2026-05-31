import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, 
    w_ptr, 
    bias_ptr, 
    out_ptr, 
    B, C, H, W, 
    OH, OW, 
    S, P, 
    K: tl.constexpr, 
    BLOCK_OH: tl.constexpr, 
    BLOCK_OW: tl.constexpr,
):
    # Program IDs
    pid_bc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    # Batch and Channel indices
    b = pid_bc // C
    c = pid_bc % C

    # Output spatial offsets
    oh_start = pid_oh * BLOCK_OH
    ow_start = pid_ow * BLOCK_OW
    
    # Create ranges for output tiles
    oh = oh_start + tl.arange(0, BLOCK_OH)
    ow = ow_start + tl.arange(0, BLOCK_OW)
    
    # Reshape to (BLOCK_OH, 1) and (1, BLOCK_OW) for broadcasting
    oh = oh[:, None]
    ow = ow[None, :]
    
    # Mask for output boundaries
    mask_out = (oh < OH) & (ow < OW)
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.float32)
    
    # Pointers to the start of the current channel for input, weight, and output
    # Input: x[b, c, h, w]
    x_channel_ptr = x_ptr + b * (C * H * W) + c * (H * W)
    # Weight: w[c, 0, kh, kw]
    w_channel_ptr = w_ptr + c * (K * K)
    # Output: out[b, c, oh, ow]
    out_channel_ptr = out_ptr + b * (C * OH * OW) + c * (OH * OW)
    
    # Convolution loop over kernel dimensions
    for kh in range(K):
        for kw in range(K):
            # Calculate input coordinates
            ih = oh * S + kh - P
            iw = ow * S + kw - P
            
            # Mask for input boundaries
            mask_in = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            
            # Load input values for the current kernel offset
            # Input offset: ih * W + iw
            x_vals = tl.load(x_channel_ptr + ih * W + iw, mask=mask_in, other=0.0)
            
            # Load weight value for the current kernel offset
            w_val = tl.load(w_channel_ptr + kh * K + kw)
            
            # Multiply-accumulate
            acc += x_vals * w_val
            
    # Add bias if it exists
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c)
        acc += bias_val
        
    # Store the result in the output tensor
    tl.store(out_channel_ptr + oh * OW + ow, acc, mask=mask_out)


def triton_depthwise_conv2d(x, weight, bias, stride=1, padding=0):
    # Ensure inputs are contiguous and on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, C, H, W = x.shape
    K = weight.shape[2]
    
    # Calculate output dimensions
    OH = (H + 2 * padding - K) // stride + 1
    OW = (W + 2 * padding - K) // stride + 1
    
    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
    
    BLOCK_OH = 32
    BLOCK_OW = 32
    
    # Grid: (B * C, ceil(OH/BLOCK_OH), ceil(OW/BLOCK_OW))
    grid = (B * C, triton.cdiv(OH, BLOCK_OH), triton.cdiv(OW, BLOCK_OW))
    
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out, 
        B, C, H, W, 
        OH, OW, 
        stride, padding, 
        K=K, 
        BLOCK_OH=BLOCK_OH, 
        BLOCK_OW=BLOCK_OW
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the Conv2d layer to manage weights and bias
        self.conv2d = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels, 
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the Conv2d layer
        weight = self.conv2d.weight
        bias = self.conv2d.bias
        stride = self.conv2d.stride[0]
        padding = self.conv2d.padding[0]
        
        # Call the Triton-optimized implementation
        return triton_depthwise_conv2d(x, weight, bias, stride, padding)