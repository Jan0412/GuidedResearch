import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool3d_kernel(
    x_ptr, out_ptr,
    B, C, D, H, W,
    Do, Ho, Wo,
    k_size, stride, padding,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_W: tl.constexpr,
):
    # Grid: (B * C * Do * Ho, (Wo + BLOCK_W - 1) // BLOCK_W)
    pid_0 = tl.program_id(0)
    pid_w = tl.program_id(1)

    # Decompose pid_0 into batch, channel, depth_out, height_out
    # pid_0 = b * (C * Do * Ho) + c * (Do * Ho) + d_out * Ho + h_out
    b = pid_0 // (C * Do * Ho)
    rem = pid_0 % (C * Do * Ho)
    c = rem // (Do * Ho)
    rem = rem % (Do * Ho)
    d_out = rem // Ho
    h_out = rem % Ho

    # Width offsets for the current block
    w_out_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = w_out_offsets < Wo

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # We use while loops because k_size is not a constexpr
    kd = 0
    while kd < k_size:
        d_in = d_out * stride + kd - padding
        if d_in >= 0 and d_in < D:
            d_off = d_in * stride_d
            kh = 0
            while kh < k_size:
                h_in = h_out * stride + kh - padding
                if h_in >= 0 and h_in < H:
                    h_off = h_in * stride_h
                    kw = 0
                    while kw < k_size:
                        w_in_offsets = w_out_offsets * stride + kw - padding
                        mask_win = mask_w & (w_in_offsets >= 0) & (w_in_offsets < W)
                        
                        # Calculate pointer for the current window element
                        # Pointer arithmetic: base + b*sb + c*sc + d*sd + h*sh + w*sw
                        ptr = x_ptr + b * stride_b + c * stride_c + d_off + h_off + w_in_offsets * stride_w
                        acc += tl.load(ptr, mask=mask_win, other=0.0)
                        kw += 1
                kh += 1
        kd += 1

    # Average pooling: divide by the total volume of the kernel
    acc = acc / (k_size * k_size * k_size)
    
    # Store result to output tensor
    out_ptr_base = out_ptr + b * out_stride_b + c * out_stride_c + d_out * out_stride_d + h_out * out_stride_h + w_out_offsets * out_stride_w
    tl.store(out_ptr_base, acc, mask=mask_w)

def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    B, C, D, H, W = x.shape
    
    # Calculate output dimensions
    Do = (D + 2 * padding - kernel_size) // stride + 1
    Ho = (H + 2 * padding - kernel_size) // stride + 1
    Wo = (W + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty((B, C, Do, Ho, Wo), device=x.device, dtype=x.dtype)
    
    # Get strides for input and output
    stride_b, stride_c, stride_d, stride_h, stride_w = x.stride()
    out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w = out.stride()
    
    BLOCK_W = 32
    grid = (B * C * Do * Ho, (Wo + BLOCK_W - 1) // BLOCK_W)
    
    avg_pool3d_kernel[grid](
        x, out,
        B, C, D, H, W,
        Do, Ho, Wo,
        kernel_size, stride, padding,
        stride_b, stride_c, stride_d, stride_h, stride_w,
        out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        BLOCK_W=BLOCK_W,
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        # If stride is None, PyTorch's AvgPool3d defaults it to kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)