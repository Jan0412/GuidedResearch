import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool3d_kernel(
    x_ptr, 
    out_ptr,
    N, C, D, H, W,
    D_out, H_out, W_out,
    stride, padding, kernel_size,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_W: tl.constexpr,
):
    # Program ID for the batch, channel, depth, and height dimensions
    pid_0 = tl.program_id(0)
    # Program ID for the width dimension block
    pid_1 = tl.program_id(1)

    # Decompose pid_0 to get (n, c, d, h)
    # pid_0 = n * (C * D_out * H_out) + c * (D_out * H_out) + d * H_out + h
    n = pid_0 // (C * D_out * H_out)
    rem = pid_0 % (C * D_out * H_out)
    c = rem // (D_out * H_out)
    rem = rem % (D_out * H_out)
    d = rem // H_out
    h = rem % H_out

    # Width offsets for this block
    w_offsets = pid_1 * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offsets < W_out

    # Accumulator for the average pooling sum
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over the kernel window
    # kernel_size is passed as a constexpr, allowing the loop to be unrolled
    for kd in range(kernel_size):
        id_val = d * stride + kd - padding
        if id_val >= 0 and id_val < D:
            for kh in range(kernel_size):
                ih_val = h * stride + kh - padding
                if ih_val >= 0 and ih_val < H:
                    for kw in range(kernel_size):
                        iw_val = w_offsets * stride + kw - padding
                        # Mask for width boundary and the block boundary
                        iw_mask = (iw_val >= 0) & (iw_val < W) & w_mask
                        
                        # Calculate input pointer offset
                        # Pointer = base + n*sn + c*sc + id*sd + ih*sh + iw*sw
                        idx = (n * stride_n +
                               c * stride_c +
                               id_val * stride_d +
                               ih_val * stride_h +
                               iw_val * stride_w)
                        
                        val = tl.load(x_ptr + idx, mask=iw_mask, other=0.0)
                        acc += val

    # Average the sum by dividing by the total number of elements in the kernel window
    # PyTorch AvgPool3d default count_include_pad=True
    out_val = acc / (kernel_size * kernel_size * kernel_size)
    
    # Calculate output pointer offset
    out_idx = (n * out_stride_n +
               c * out_stride_c +
               d * out_stride_d +
               h * out_stride_h +
               w_offsets * out_stride_w)
    
    tl.store(out_ptr + out_idx, out_val, mask=w_mask)


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    # Ensure input is contiguous on GPU
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    N, C, D, H, W = x.shape
    
    # Calculate output dimensions
    D_out = (D + 2 * padding - kernel_size) // stride + 1
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Get strides for input and output tensors
    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w = out.stride()
    
    BLOCK_W = 16
    # Grid: (N * C * D_out * H_out, ceil(W_out / BLOCK_W))
    grid = (N * C * D_out * H_out, (W_out + BLOCK_W - 1) // BLOCK_W)
    
    avg_pool3d_kernel[grid](
        x, out,
        N, C, D, H, W,
        D_out, H_out, W_out,
        stride, padding, kernel_size,
        stride_n, stride_c, stride_d, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
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