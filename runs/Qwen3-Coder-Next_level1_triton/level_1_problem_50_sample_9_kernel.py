import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor: [B, C, H, W]
    w_ptr,  # Weight tensor: [OutC, C, KH, KW]
    out_ptr,  # Output tensor: [B, OutC, OH, OW]
    B, C, H, W,  # Input dimensions
    OutC, KH, KW,  # Weight dimensions
    stride_h, stride_w,  # Convolution strides
    pad_h, pad_w,  # Padding
    OH, OW,  # Output dimensions
    BLOCK_B: tl.constexpr,
    BLOCK_OUTC: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    
    # Compute output spatial positions
    start_oh = tl.program_id(2) * BLOCK_KH
    oh_range = start_oh + tl.arange(0, BLOCK_KH)
    oh_mask = oh_range < OH
    
    start_ow = tl.program_id(3) * BLOCK_KW
    ow_range = start_ow + tl.arange(0, BLOCK_KW)
    ow_mask = ow_range < OW
    
    # Compute the base pointers for this program
    # For input: [batch_idx, :, oh*stride_h - pad_h, ow*stride_w - pad_w]
    # But we need to handle boundary conditions
    
    # Initialize accumulator for output
    out = tl.zeros((BLOCK_KH, BLOCK_KW), dtype=tl.float32)
    
    # Iterate over input channels
    for c in range(0, C, BLOCK_C):
        c_range = c + tl.arange(0, BLOCK_C)
        c_mask = c_range < C
        
        # Iterate over kernel height
        for kh in range(0, KH, BLOCK_KH):
            kh_range = kh + tl.arange(0, BLOCK_KH)
            kh_mask = kh_range < KH
            
            # Iterate over kernel width
            for kw in range(0, KW, BLOCK_KW):
                kw_range = kw + tl.arange(0, BLOCK_KW)
                kw_mask = kw_range < KW
                
                # Load kernel weights [BLOCK_OUTC, BLOCK_C, BLOCK_KH, BLOCK_KW]
                # But we're only doing one out_c_idx, so we'll load a subset
                
                # Compute input offsets
                oh_offsets = oh_range[:, None] * stride_h - pad_h + kh_range[None, :]
                ow_offsets = ow_range[None, :] * stride_w - pad_w + kw_range[None, :]
                
                # Create masks for valid positions
                h_valid = (oh_offsets >= 0) & (oh_offsets < H)
                w_valid = (ow_offsets >= 0) & (ow_offsets < W)
                valid_mask = h_valid & w_valid
                
                # Load input values: [BLOCK_KH, BLOCK_KW, BLOCK_C]
                input_ptrs = x_ptr + batch_idx * (C * H * W) + c_range[None, None, :] * (H * W) + \
                            oh_offsets[:, :, None] * W + ow_offsets[:, :, None]
                
                input_vals = tl.load(input_ptrs, mask=valid_mask[:, :, None] & c_mask[None, None, :], other=0.0)
                
                # Load kernel weights: [BLOCK_C, BLOCK_KH, BLOCK_KW]
                weight_ptrs = w_ptr + out_c_idx * (C * KH * KW) + c_range[:, None, None] * (KH * KW) + \
                             kh_range[None, :, None] * KW + kw_range[None, None, :]
                
                weight_vals = tl.load(weight_ptrs, mask=c_mask[:, None, None] & kh_mask[None, :, None] & kw_mask[None, None, :], other=0.0)
                
                # Compute partial dot product
                # input_vals: [BLOCK_KH, BLOCK_KW, BLOCK_C]
                # weight_vals: [BLOCK_C, BLOCK_KH, BLOCK_KW]
                # Result should be [BLOCK_KH, BLOCK_KW]
                partial_out = tl.sum(input_vals * weight_vals[None, :, :, :], axis=2)
                out += partial_out.to(tl.float32)
    
    # Store result
    out_ptrs = out_ptr + batch_idx * (OutC * OH * OW) + out_c_idx * (OH * OW) + \
              oh_range[:, None] * OW + ow_range[None, :]
    tl.store(out_ptrs, out, mask=oh_mask[:, None] & ow_mask[None, :])


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, stride=4, padding=2):
    """
    Triton implementation of Conv2d for the specific case in the model.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C, H, W = x.shape
    OutC, _, KH, KW = weight.shape
    
    # Calculate output dimensions
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    
    # Create output tensor
    out = torch.empty(B, OutC, OH, OW, device=x.device, dtype=x.dtype)
    
    # Define block sizes - tuned for the specific case
    BLOCK_B = 1
    BLOCK_OUTC = 1
    BLOCK_C = 32
    BLOCK_KH = 4
    BLOCK_KW = 4
    
    # Grid dimensions: [batch, out_channels, oh_blocks, ow_blocks]
    grid = lambda meta: (
        B,
        OutC,
        (OH + meta["BLOCK_KH"] - 1) // meta["BLOCK_KH"],
        (OW + meta["BLOCK_KW"] - 1) // meta["BLOCK_KW"]
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, out,
        B, C, H, W,
        OutC, KH, KW,
        stride, stride,
        padding, padding,
        OH, OW,
        BLOCK_B=BLOCK_B,
        BLOCK_OUTC=BLOCK_OUTC,
        BLOCK_C=BLOCK_C,
        BLOCK_KH=BLOCK_KH,
        BLOCK_KW=BLOCK_KW,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        # Use Triton convolution instead of PyTorch's native implementation
        return triton_conv2d(x, self.conv1.weight, stride=4, padding=2) + self.conv1.bias.view(1, -1, 1, 1)