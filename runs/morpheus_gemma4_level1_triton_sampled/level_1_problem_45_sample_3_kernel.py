import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr, out_ptr,
    B, C, H, W,
    OH, OW,
    stride_h, stride_w,
    pad_h, pad_w,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    # Each program handles a block of output elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (B * C * OH * OW)

    # Decompose flat index to 4D coordinates (b, c, oh, ow)
    ow = offsets % OW
    oh = (offsets // OW) % OH
    bc = offsets // (OH * OW)
    b = bc // C
    c = bc % C

    # Accumulator for the window sum
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the pooling window
    for kh in range(0, KH):
        h_idx = oh * stride_h - pad_h + kh
        h_mask = (h_idx >= 0) & (h_idx < H)
        for kw in range(0, KW):
            w_idx = ow * stride_w - pad_w + kw
            w_mask = (w_idx >= 0) & (w_idx < W)

            # Compute pointer to the input element
            # x_ptr + b*stride_xb + c*stride_xc + h_idx*stride_xh + w_idx*stride_xw
            ptr = x_ptr + b * stride_xb + c * stride_xc + h_idx * stride_xh + w_idx * stride_xw
            
            # Mask for boundary checks and block size
            load_mask = mask & h_mask & w_mask
            val = tl.load(ptr, mask=load_mask, other=0.0)
            acc += val

    # Compute average
    out = acc / (KH * KW)

    # Compute pointer to the output element and store
    out_ptr_final = out_ptr + b * stride_ob + c * stride_oc + oh * stride_oh + ow * stride_ow
    tl.store(out_ptr_final, out, mask=mask)


def triton_avg_pool(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Wrapper for the Triton Average Pooling kernel.
    """
    B, C, H, W = x.shape
    
    # Handle default stride
    sh = stride if stride is not None else kernel_size
    sw = stride if stride is not None else kernel_size
    ph, pw = padding, padding
    
    # Calculate output dimensions
    OH = (H + 2 * ph - kernel_size) // sh + 1
    OW = (W + 2 * pw - kernel_size) // sw + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
    
    # Get strides for 4D tensor indexing
    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_ob, stride_oc, stride_oh, stride_ow = out.stride()
    
    BLOCK_SIZE = 1024
    # Grid is based on the total number of output elements
    grid = lambda meta: ((B * C * OH * OW + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    avg_pool_kernel[grid](
        x, out,
        B, C, H, W,
        OH, OW,
        sh, sw,
        ph, pw,
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_ob, stride_oc, stride_oh, stride_ow,
        KH=kernel_size, KW=kernel_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        # Ensure input is FP32 and on CUDA as required by the kernel
        x = x.to(torch.float32)
        return triton_avg_pool(x, self.kernel_size, self.stride, self.padding)