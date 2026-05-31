import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    out_ptr, in_ptr, weight_ptr, bias_ptr,
    B, IC, OC, H, W, D, K, S, P,
    H_out, W_out, D_out,
    groups,
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Grid: each block computes one output element
    pid = tl.program_id(0)
    
    # Decode output coordinates
    # Output layout: (B, OC, H_out, W_out, D_out)
    # Strides: OC -> H_out*W_out*D_out, H_out -> W_out*D_out, W_out -> D_out, D_out -> 1
    stride_oc = H_out * W_out * D_out
    stride_h = W_out * D_out
    stride_w = D_out
    
    oc = (pid // (B * stride_oc)) % OC
    h = (pid // (B * stride_h)) % H_out
    w = (pid // (B * stride_w)) % W_out
    d = pid % D_out
    b = pid // (B * stride_oc * OC)
    
    # Check bounds
    if b >= B or oc >= OC or h >= H_out or w >= W_out or d >= D_out:
        return

    # Input base index for this output element
    # Input layout: (B, IC, H, W, D)
    # Strides: IC -> H*W*D, H -> W*D, W -> D, D -> 1
    stride_ic_in = H * W * D
    stride_h_in = W * D
    stride_w_in = D
    
    # Effective start position in input for this output pixel
    # account for stride and padding
    start_h = h * S - P
    start_w = w * S - P
    
    # Base index in input tensor for the top-left of the window
    # We will add offsets for kx, ky, ic
    base_idx_in = b * stride_ic_in * IC + start_h * stride_h_in + start_w * stride_w_in + d
    
    # Weight base index
    # Weight layout: (OC, IC, K, K, 1)
    # Strides: IC -> K*K, K -> K, K -> 1, 1 -> 0 (implicit)
    # Actually layout is (OC, IC, K, K, 1)
    # Stride of IC is K*K. Stride of Kx is K. Stride of Ky is 1.
    # Weight index: oc * (IC * K * K) + ic * (K * K) + kx * K + ky
    stride_ic_w = K * K
    stride_kx = K
    stride_ky = 1
    
    base_idx_w = oc * IC * stride_ic_w
    
    # Accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Groups handling
    OC_per_group = OC // groups
    IC_per_group = IC // groups
    oc_group = oc // OC_per_group
    
    # Tiling over input channels
    for ic_off in range(0, IC, BLOCK_SIZE_IC):
        ic_offsets = ic_off + tl.arange(0, BLOCK_SIZE_IC)
        mask_ic = ic_offsets < IC
        
        # Group check mask
        ic_group = ic_offsets // IC_per_group
        mask_group = ic_group == oc_group
        
        # Load weight tile
        # Weight shape: (BLOCK_SIZE_IC, K, K)
        # Indices: base_idx_w + ic_offsets * stride_ic_w
        # We need to broadcast kx, ky
        kx_offsets = tl.arange(0, BLOCK_SIZE_K)
        ky_offsets = tl.arange(0, BLOCK_SIZE_K)
        
        # Create grid for kx, ky
        # kx_grid: (BLOCK_SIZE_K, 1), ky_grid: (1, BLOCK_SIZE_K)
        # But tl.load expects 1D or broadcastable.
        # We can compute indices directly.
        # Weight indices: base_idx_w + ic_offsets[:, None] * stride_ic_w + kx_grid[None, :] * stride_kx + ky_grid[None, :] * stride_ky
        # Triton handles broadcasting in arithmetic if shapes match.
        # ic_offsets is (BLOCK_SIZE_IC,), kx is (BLOCK_SIZE_K,), ky is (BLOCK_SIZE_K,)
        # We want tile shape (BLOCK_SIZE_IC, BLOCK_SIZE_K, BLOCK_SIZE_K)
        # This might be too large if BLOCK_SIZE_K is not small.
        # K is small (e.g., 3), so BLOCK_SIZE_K should be K.
        # Let's assume BLOCK_SIZE_K == K.
        
        # Weight indices
        w_idx = base_idx_w + ic_offsets[:, None, None] * stride_ic_w + kx_offsets[None, :, None] * stride_kx + ky_offsets[None, None, :] * stride_ky
        mask_w = mask_ic[:, None, None]
        
        # Load weights
        w_tile = tl.load(weight_ptr + w_idx, mask=mask_w, other=0.0)
        
        # Input indices
        # Input shape: (BLOCK_SIZE_IC, K, K)
        # Indices: base_idx_in + ic_offsets * stride_ic_in + kx * stride_h_in + ky * stride_w_in
        # kx, ky are spatial offsets relative to start_h, start_w
        # We need to mask out of bounds spatial accesses
        # Actual input coordinates: start_h + kx, start_w + ky
        # Valid if 0 <= start_h + kx < H and 0 <= start_w + ky < W
        
        # Create masks for spatial bounds
        # kx_offsets: (BLOCK_SIZE_K,)
        # ky_offsets: (BLOCK_SIZE_K,)
        # We need mask of shape (BLOCK_SIZE_K, BLOCK_SIZE_K)
        # mask_h: (BLOCK_SIZE_K, 1), mask_w: (1, BLOCK_SIZE_K)
        mask_h = (start_h + kx_offsets[:, None]) >= 0
        mask_h = mask_h & ((start_h + kx_offsets[:, None]) < H)
        mask_w = (start_w + ky_offsets[None, :]) >= 0
        mask_w = mask_w & ((start_w + ky_offsets[None, :]) < W)
        
        # Combine spatial masks
        mask_spatial = mask_h & mask_w
        
        # Input indices
        i_idx = base_idx_in + ic_offsets[:, None, None] * stride_ic_in + kx_offsets[None, :, None] * stride_h_in + ky_offsets[None, None, :] * stride_w_in
        mask_i = mask_ic[:, None, None] & mask_spatial[None, :, :]
        
        # Load input
        i_tile = tl.load(in_ptr + i_idx, mask=mask_i, other=0.0)
        
        # Multiply and accumulate
        # w_tile: (BLOCK_SIZE_IC, K, K)
        # i_tile: (BLOCK_SIZE_IC, K, K)
        # mask_group: (BLOCK_SIZE_IC,) -> needs broadcast
        mask_group_bc = mask_group[:, None, None]
        
        acc += tl.sum(w_tile * i_tile * mask_group_bc)
    
    # Add bias
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + oc)
    
    # Store output
    out_idx = b * OC * stride_oc + oc * stride_oc + h * stride_h + w * stride_w + d
    tl.store(out_ptr + out_idx, acc)


def triton_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                  stride: int, padding: int, groups: int) -> torch.Tensor:
    """
    Triton implementation of 3D convolution.
    Assumes kernel depth is 1.
    """
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    B, IC, H, W, D = x.shape
    OC, IC_w, K, K_w, K_d = weight.shape
    
    assert IC == IC_w, "Input channels must match weight input channels"
    assert K == K_w, "Kernel height must match kernel width"
    assert K_d == 1, "Kernel depth must be 1"
    
    # Output dimensions
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1
    D_out = (D + 2 * 0 - 1) // stride + 1  # Padding applies to D too, but Kd=1, so D_out=D if padding=0
    # Actually padding applies to all dims.
    # D_out = (D + 2*padding - 1) // stride + 1
    # Since Kd=1, D_out = D if padding=0.
    # General formula:
    D_out = (D + 2 * padding - 1) // stride + 1
    
    out = torch.empty((B, OC, H_out, W_out, D_out), device=x.device, dtype=x.dtype)
    
    # Tunable parameters
    BLOCK_SIZE_IC = 64
    BLOCK_SIZE_K = K  # K is small, so tile size equals K
    
    # Grid size: total output elements
    n_elements = B * OC * H_out * W_out * D_out
    
    grid = (n_elements,)
    
    conv3d_kernel[grid](
        out, x, weight, bias,
        B, IC, OC, H, W, D, K, stride, padding,
        H_out, W_out, D_out,
        groups,
        BLOCK_SIZE_IC=BLOCK_SIZE_IC,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters for the forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias manually or use a dummy conv to get shape
        # Since we are replacing the operator, we need the weights.
        # We can use nn.Conv3d to initialize, then detach, or initialize manually.
        # Using nn.Conv3d for initialization is safe.
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        
        # Initialize weights and bias tensors
        self.weight = nn.Parameter(self.conv3d.weight.data.clone())
        if bias:
            self.bias = nn.Parameter(self.conv3d.bias.data.clone())
        else:
            self.bias = None
            
        # Delete the nn.Conv3d module as we use custom kernel
        del self.conv3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(x, self.weight, self.bias, self.stride, self.padding, self.groups)


def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    width = 256
    height = 256
    depth = 10
    
    x = torch.rand(batch_size, in_channels, height, width, depth, device='cuda')
    return [x]


def get_init_inputs():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]