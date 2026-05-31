import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool_kernel(
    x_ptr, 
    out_ptr,
    batch, channels, h, w,
    ho, wo,
    k_size, stride, padding, dilation,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_n_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Decompose pid_n_c into batch index and channel index
    n = pid_n_c // channels
    c = pid_n_c % channels

    # Calculate the range of output pixels this program handles
    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W
    
    oh_offsets = oh_start + tl.arange(0, BLOCK_H)
    ow_offsets = ow_start + tl.arange(0, BLOCK_W)

    # Boundary masks for output
    mask_h = oh_offsets < ho
    mask_w = ow_offsets < wo
    out_mask = mask_h[:, None] & mask_w[None, :]

    # Initialize max values to a very small number (effectively -infinity)
    max_vals = tl.full((BLOCK_H, BLOCK_W), -1e38, dtype=tl.float32)

    # Iterate over the pooling window
    for i in range(k_size):
        for j in range(k_size):
            # Calculate input coordinates for the current window element
            # h_in = oh * stride - padding + i * dilation
            # w_in = ow * stride - padding + j * dilation
            h_in = oh_offsets[:, None] * stride - padding + i * dilation
            w_in = ow_offsets[None, :] * stride - padding + j * dilation

            # Boundary masks for input (padding handled by 'other' value in tl.load)
            mask_in_h = (h_in >= 0) & (h_in < h)
            mask_in_w = (w_in >= 0) & (w_in < w)
            mask_in = mask_in_h & mask_in_w

            # Compute pointer for the input element
            # Pointer = x_ptr + n*stride_n + c*stride_c + h_in*stride_h + w_in*stride_w
            ptr = x_ptr + n * stride_xn + c * stride_xc + h_in * stride_xh + w_in * stride_xw
            
            # Load input value or -inf if out of bounds
            vals = tl.load(ptr, mask=mask_in, other=-1e38)
            
            # Update max
            max_vals = tl.maximum(max_vals, vals)

    # Store the result to output tensor
    out_ptr_base = out_ptr + n * stride_on + c * stride_oc
    out_offsets = oh_offsets[:, None] * stride_oh + ow_offsets[None, :] * stride_ow
    tl.store(out_ptr_base + out_offsets, max_vals, mask=out_mask)


def triton_maxpool2d(x, kernel_size, stride, padding, dilation):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    n, c, h, w = x.shape
    
    # Calculate output dimensions
    ho = (h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    wo = (w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((n, c, ho, wo), device=x.device, dtype=x.dtype)
    
    # Strides for input and output
    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_on, stride_oc, stride_oh, stride_ow = out.stride()
    
    BLOCK_H = 16
    BLOCK_W = 16
    
    grid = (n * c, (ho + BLOCK_H - 1) // BLOCK_H, (wo + BLOCK_W - 1) // BLOCK_W)
    
    maxpool_kernel[grid](
        x, out,
        n, c, h, w,
        ho, wo,
        kernel_size, stride, padding, dilation,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_on, stride_oc, stride_oh, stride_ow,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor using the Triton implementation.
        """
        # Ensure input is float32 as requested
        x = x.to(torch.float32)
        return triton_maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)