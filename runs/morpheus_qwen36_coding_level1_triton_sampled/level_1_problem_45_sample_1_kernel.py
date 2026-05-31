import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_kernel(
    x_ptr, out_ptr,
    x_stride_b, x_stride_c, x_stride_h, W_in,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    stride, kernel_size,
    num_rows: tl.constexpr,
    row_len: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    
    # Compute base input pointer for the current block
    input_ptr = x_ptr + b * x_stride_b + c * x_stride_c + i * x_stride_h + j * stride
    
    # Compute output pointer
    out_ptr_idx = out_ptr + b * out_stride_b + c * out_stride_c + i * out_stride_h + j
    
    sum_val = 0.0
    
    # Iterate over rows within the kernel window
    for r in range(num_rows):
        # Pointer to the current row
        row_ptr = input_ptr + r * x_stride_h
        
        # Load the row elements
        offsets = tl.arange(0, row_len)
        mask = offsets < W_in
        vals = tl.load(row_ptr + offsets, mask=mask, other=0.0)
        
        # Accumulate sum
        sum_val += tl.sum(vals)
        
    # Compute average
    out = sum_val / (num_rows * row_len)
    
    # Store result
    tl.store(out_ptr_idx, out)


def triton_avg_pool(x: torch.Tensor, kernel_size: int = 11, stride: int = None, padding: int = 0) -> torch.Tensor:
    assert padding == 0, "Padding is not supported in this optimized kernel."
    if stride is None:
        stride = kernel_size
        
    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Compute strides
    x_stride_b = C * H * W
    x_stride_c = H * W
    x_stride_h = W
    # W_in is passed for masking
    
    out_stride_b = C * H_out * W_out
    out_stride_c = H_out * W_out
    out_stride_h = W_out
    out_stride_w = 1
    
    grid = (B, C, H_out, W_out)
    
    avg_pool_kernel[grid](
        x, out,
        x_stride_b, x_stride_c, x_stride_h, W,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        stride, kernel_size,
        num_rows=kernel_size,
        row_len=kernel_size,
        num_warps=4
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)