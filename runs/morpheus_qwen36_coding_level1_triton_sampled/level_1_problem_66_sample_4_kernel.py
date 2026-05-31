import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    groups,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    
    # Calculate output channel offset
    oc_offset = pid_b * OC + pid_oc
    
    # Determine group for this output channel
    group_size = OC // groups
    group_id = oc_offset // group_size
    
    # Input channel size per group
    ic_per_group = IC // groups
    
    # Total reduction size K
    K = ic_per_group * KD * KH * KW
    
    # Base pointers for weight and input
    w_base = w_ptr + oc_offset * (ic_per_group * KD * KH * KW)
    x_base = x_ptr + pid_b * (IC * ID * IH * IW)
    
    # Spatial output coordinates
    spatial_offsets = tl.arange(0, BLOCK_SIZE_SPATIAL)
    od = spatial_offsets // (OH * OW)
    rem = spatial_offsets % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    
    # Mask for valid spatial coordinates
    mask_sp = (od < OD) & (oh < OH) & (ow < OW)
    
    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_SPATIAL,), dtype=tl.float32)
    
    # Loop over reduction dimension K
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        # Decompose k into (i, kd, kh, kw)
        # i is within [0, ic_per_group)
        # kd, kh, kw are kernel indices
        k_idx = k_offsets
        kw = k_idx % KW
        k_idx //= KW
        kh = k_idx % KH
        k_idx //= KH
        kd = k_idx % KD
        k_idx //= KD
        i = k_idx
        
        # Input channel index considering groups
        i_global = i + group_id * ic_per_group
        
        # Input spatial coordinates with dilation and padding
        # di = od * stride_d + kd * dilation_d - padding_d
        di = od * stride_d + kd * dilation_d - padding_d
        dh = oh * stride_h + kh * dilation_h - padding_h
        dw = ow * stride_w + kw * dilation_w - padding_w
        
        # Mask for valid input coordinates
        mask_input = k_mask & (di >= 0) & (di < ID) & (dh >= 0) & (dh < IH) & (dw >= 0) & (dw < IW)
        
        # Load weights
        w_ptr_offsets = w_base + i * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
        w = tl.load(w_ptr_offsets, mask=k_mask, other=0.0)
        
        # Load inputs
        x_ptr_offsets = x_base + i_global * (ID * IH * IW) + di * (IH * IW) + dh * IW + dw
        x = tl.load(x_ptr_offsets, mask=mask_input, other=0.0)
        
        # Accumulate
        acc += tl.where(mask_input, w * x, 0.0)
    
    # Add bias if available (b_ptr is not None, but we assume b_ptr is passed as valid pointer or we handle in wrapper)
    # For simplicity, we assume bias is added outside or b_ptr is handled. 
    # Here we add bias using the output channel index.
    # Note: b_ptr should be passed. We'll handle bias in the wrapper or assume b_ptr is valid.
    # Let's assume b_ptr is passed and valid.
    b = tl.load(b_ptr + oc_offset, mask=mask_sp, other=0.0)
    acc += b
    
    # Store output
    out_ptr_offsets = out_ptr + pid_b * (OC * OD * OH * OW) + oc_offset * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow
    tl.store(out_ptr_offsets, acc, mask=mask_sp)


def triton_conv3d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, 
                  stride: tuple, padding: tuple, dilation: tuple, groups: int) -> torch.Tensor:
    B, IC, ID, IH, IW = x.shape
    OC, IC_w, KD, KH, KW = w.shape
    assert IC_w == IC // groups, "Input channels must be divisible by groups and match weight channels"
    
    OD = (ID + 2 * padding[0] - dilation[0] * (KD - 1) - 1) // stride[0] + 1
    OH = (IH + 2 * padding[1] - dilation[1] * (KH - 1) - 1) // stride[1] + 1
    OW = (IW + 2 * padding[2] - dilation[2] * (KW - 1) - 1) // stride[2] + 1
    
    out = torch.empty((B, OC, OD, OH, OW), dtype=x.dtype, device=x.device)
    
    # Tunable parameters
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_OC = 1
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_SPATIAL = 64
    
    grid = (B, OC)
    
    conv3d_kernel[grid](
        x, w, b, out,
        B, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        groups,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_SPATIAL=BLOCK_SIZE_SPATIAL,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return triton_conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            # Handle bias=False by passing a dummy or handling in kernel. 
            # For simplicity, we assume bias is always present or handle in wrapper.
            # Here we create a zero tensor for bias if not present.
            b_dummy = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
            return triton_conv3d(x, self.weight, b_dummy, self.stride, self.padding, self.dilation, self.groups)