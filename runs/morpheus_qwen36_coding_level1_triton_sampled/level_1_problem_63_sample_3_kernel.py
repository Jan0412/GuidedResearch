import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    stride, padding, dilation, kernel_size,
    in_channels, out_channels, height, width,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr, HAS_BIAS: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Output tile coordinates
    c_offsets = pid_c * BLOCK_M + tl.arange(0, BLOCK_M)
    h_offsets = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
    w_offsets = pid_w * BLOCK_P + tl.arange(0, BLOCK_P)
    
    # Input channel offsets
    k_offsets = tl.arange(0, BLOCK_K)
    
    # Kernel offsets
    kh_offsets = tl.arange(0, kernel_size)
    kw_offsets = tl.arange(0, kernel_size)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_P), dtype=tl.float32)
    
    # Precompute input spatial bounds
    h_in_base = h_offsets * stride - padding
    w_in_base = w_offsets * stride - padding
    
    # Loop over input channels
    for k_start in range(0, in_channels, BLOCK_K):
        k_idx = k_start + k_offsets[:, None, None, None, None]
        
        # Generate weight indices
        w_c = c_offsets[:, None, None, None, None]
        w_k = k_idx
        w_kh = kh_offsets[None, None, :, None, None]
        w_kw = kw_offsets[None, None, None, :, None]
        
        w_idx = (w_c * in_channels + w_k) * kernel_size * kernel_size + w_kh * kernel_size + w_kw
        w_ptr_tile = w_ptr + w_idx
        w = tl.load(w_ptr_tile)
        
        # Generate input indices
        # h_in = h_in_base + kh * dilation
        # w_in = w_in_base + kw * dilation
        h_in = h_in_base[:, None, None, None] + kh_offsets[None, None, :, None] * dilation
        w_in = w_in_base[None, :, None, None] + kw_offsets[None, None, None, :] * dilation
        
        # Reshape for broadcasting
        h_in = h_in[:, :, None, :, None]
        w_in = w_in[:, :, None, :, None]
        
        # Masking for input bounds
        mask_h = (h_in >= 0) & (h_in < height)
        mask_w = (w_in >= 0) & (w_in < width)
        mask = mask_h & mask_w
        
        # Input indices
        x_idx = (pid_n * in_channels + k_idx) * height * width + h_in * width + w_in
        x_ptr_tile = x_ptr + x_idx
        x = tl.load(x_ptr_tile, mask=mask, other=0.0)
        
        # Reshape for dot product
        # w: (BLOCK_M, BLOCK_K, kernel_size, kernel_size) -> (BLOCK_M, BLOCK_K * kernel_size * kernel_size)
        w_reshaped = tl.reshape(w, (BLOCK_M, BLOCK_K * kernel_size * kernel_size))
        # x: (BLOCK_K, kernel_size, kernel_size, BLOCK_N, BLOCK_P) -> (BLOCK_K * kernel_size * kernel_size, BLOCK_N * BLOCK_P)
        x_reshaped = tl.reshape(x, (BLOCK_K * kernel_size * kernel_size, BLOCK_N * BLOCK_P))
        
        # Accumulate
        acc += tl.dot(w_reshaped, x_reshaped)
    
    # Reshape acc to (BLOCK_M, BLOCK_N * BLOCK_P) for storage
    acc = tl.reshape(acc, (BLOCK_M, BLOCK_N * BLOCK_P))
    
    # Store result
    out_idx = (pid_n * out_channels + c_offsets[:, None]) * (height * width) + h_offsets[None, :] * width + w_offsets[None, :]
    out_ptr_tile = out_ptr + out_idx
    tl.store(out_ptr_tile, acc, mask=True)
    
    # Add bias if present
    if HAS_BIAS:
        b = tl.load(b_ptr + c_offsets)
        acc = acc + b[:, None]
        tl.store(out_ptr_tile, acc, mask=True)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor = None, 
                  stride: int = 1, padding: int = 0, dilation: int = 1, 
                  kernel_size: int = 3, in_channels: int = 16, out_channels: int = 128, 
                  height: int = 1024, width: int = 1024) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()
    
    batch_size = x.shape[0]
    height_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    width_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((batch_size, out_channels, height_out, width_out), dtype=torch.float32, device=x.device)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_P = 32
    BLOCK_K = 32
    
    grid = (batch_size, tl.cdiv(out_channels, BLOCK_M), tl.cdiv(height_out, BLOCK_N), tl.cdiv(width_out, BLOCK_P))
    
    conv2d_kernel[grid](
        x, w, b, out,
        stride, padding, dilation, kernel_size,
        in_channels, out_channels, height, width,
        BLOCK_M, BLOCK_N, BLOCK_P, BLOCK_K,
        HAS_BIAS=(b is not None)
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
            self.kernel_size, self.in_channels, self.out_channels,
            x.shape[2], x.shape[3]
        )