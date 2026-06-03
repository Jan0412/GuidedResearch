import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def transposed_conv2d_kernel(
    # Pointers to input/output tensors
    x_ptr, w_ptr, out_ptr,
    # Dimensions
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_oc, w_stride_ic, w_stride_kh, w_kw,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    # Block sizes
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute the position in output
    out_h_idx = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w_idx = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Compute the corresponding input positions (with stride)
    in_h_start = out_h_idx * stride_h - pad_h
    in_w_start = out_w_idx * stride_w - pad_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(0, in_channels, BLOCK_SIZE_C):
        ic_range = ic + tl.arange(0, BLOCK_SIZE_C)
        mask_ic = ic_range < in_channels
        
        # Load input data
        in_h_idx = in_h_start[:, None] + tl.arange(0, BLOCK_SIZE_KH)[None, :]
        in_w_idx = in_w_start[None, :] + tl.arange(0, BLOCK_SIZE_KW)[:, None]
        
        # Create masks for valid input positions
        mask_h = (in_h_idx >= 0) & (in_h_idx < in_h)
        mask_w = (in_w_idx >= 0) & (in_w_idx < in_w)
        mask = mask_h & mask_w & mask_ic[None, :, None]
        
        # Load input values
        x_offsets = (
            pid_b * x_stride_b +
            ic_range[None, :, None] * x_stride_c +
            in_h_idx[:, None, :] * x_stride_h +
            in_w_idx[None, :, :] * x_stride_w
        )
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Load corresponding weights
        # For transposed conv: out_c depends on kernel position relative to output
        kh_range = tl.arange(0, BLOCK_SIZE_KH)
        kw_range = tl.arange(0, BLOCK_SIZE_KW)
        
        # Calculate kernel offsets
        kh_idx = (out_h_idx[:, None] * stride_h - in_h_start[:, None])[:, None, :] + kh_range[None, :, None]
        kw_idx = (out_w_idx[None, :] * stride_w - in_w_start[None, :])[None, :, :] + kw_range[None, None, :]
        
        # Only consider kernel positions that are within kernel bounds
        mask_kh = (kh_idx >= 0) & (kh_idx < kernel_h)
        mask_kw = (kw_idx >= 0) & (kw_idx < kernel_w)
        mask_kernel = mask_kh & mask_kw & mask_ic[None, :, None]
        
        w_offsets = (
            pid_oc * w_stride_oc +
            ic_range[None, :, None] * w_stride_ic +
            kh_idx * w_stride_kh +
            kw_idx * w_kw
        )
        w = tl.load(w_ptr + w_offsets, mask=mask_kernel, other=0.0)
        
        # Accumulate: x * w
        acc += tl.sum(x * w, axis=1)
    
    # Store result
    out_offsets = (
        pid_b * out_stride_b +
        pid_oc * out_stride_c +
        out_h_idx[:, None] * out_stride_h +
        out_w_idx[None, :] * out_stride_w
    )
    tl.store(out_ptr + out_offsets, acc.to(x_ptr.dtype.element_ty))


class TritonConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights (similar to PyTorch's default initialization)
        kh, kw = self.kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kh, kw) * (2.0 / (in_channels * kh * kw)) ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels, _, kh, kw = self.weight.shape
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        
        # Calculate output dimensions (matching PyTorch's ConvTranspose2d)
        out_h = (in_h - 1) * stride_h - 2 * pad_h + kh
        out_w = (in_w - 1) * stride_w - 2 * pad_w + kw
        
        # Allocate output tensor
        out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
        
        # Define block sizes for Triton kernel (tunable parameters)
        BLOCK_SIZE_B = 1
        BLOCK_SIZE_C = min(32, in_channels)
        BLOCK_SIZE_H = 16
        BLOCK_SIZE_W = 16
        BLOCK_SIZE_KH = min(3, kh)
        BLOCK_SIZE_KW = min(7, kw)
        
        # Grid dimensions
        grid = (
            (batch_size + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B,
            (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C,
            (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
            (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
        )
        
        # Calculate strides
        x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
        w_stride_oc, w_stride_ic, w_stride_kh, w_kw = self.weight.stride()
        out_stride_b, out_stride_c, out_stride_h, out_stride_w = out.stride()
        
        # Launch kernel
        transposed_conv2d_kernel[grid](
            x, self.weight, out,
            batch_size, in_channels, out_channels,
            in_h, in_w, out_h, out_w,
            kh, kw,
            stride_h, stride_w,
            pad_h, pad_w,
            x_stride_b, x_stride_c, x_stride_h, x_stride_w,
            w_stride_oc, w_stride_ic, w_stride_kh, w_kw,
            out_stride_b, out_stride_c, out_stride_h, out_stride_w,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_KH=BLOCK_SIZE_KH,
            BLOCK_SIZE_KW=BLOCK_SIZE_KW,
        )
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        
        return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernels for 2D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = TritonConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D transposed convolution.
        """
        return self.conv_transpose2d(x)