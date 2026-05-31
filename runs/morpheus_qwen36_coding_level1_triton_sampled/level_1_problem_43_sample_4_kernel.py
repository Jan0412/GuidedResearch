import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def maxpool3d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    dim1, dim2, dim3,
    d1_out, d2_out, d3_out,
    kernel_size,
    stride,
    padding,
    dilation,
    stride_n,
    stride_c,
    stride_d1,
    stride_d2,
    BLOCK_SIZE: tl.constexpr = 1
):
    pid = tl.program_id(0)
    
    # Compute output coordinates (n, c, d1, d2, d3) from pid
    n = pid // (channels * d1_out * d2_out * d3_out)
    rem = pid % (channels * d1_out * d2_out * d3_out)
    c = rem // (d1_out * d2_out * d3_out)
    rem = rem % (d1_out * d2_out * d3_out)
    d1 = rem // (d2_out * d3_out)
    rem = rem % (d2_out * d3_out)
    d2 = rem // d3_out
    d3 = rem % d3_out
    
    # Base pointer for (n, c)
    base_ptr = input_ptr + n * stride_n + c * stride_c
    
    # Starting coordinates in input space
    start_d1 = d1 * stride - padding
    start_d2 = d2 * stride - padding
    start_d3 = d3 * stride - padding
    
    max_val = -float('inf')
    
    # Iterate over kernel volume
    for k1 in range(kernel_size):
        for k2 in range(kernel_size):
            for k3 in range(kernel_size):
                idx_d1 = start_d1 + k1 * dilation
                idx_d2 = start_d2 + k2 * dilation
                idx_d3 = start_d3 + k3 * dilation
                
                # Check bounds (handles padding implicitly)
                if idx_d1 >= 0 and idx_d1 < dim1 and idx_d2 >= 0 and idx_d2 < dim2 and idx_d3 >= 0 and idx_d3 < dim3:
                    offset = idx_d1 * stride_d1 + idx_d2 * stride_d2 + idx_d3
                    val = tl.load(base_ptr + offset)
                    if val > max_val:
                        max_val = val
                        
    tl.store(output_ptr + pid, max_val)

def triton_maxpool3d(x, kernel_size, stride, padding, dilation, ceil_mode):
    assert x.is_cuda and x.dtype == torch.float32, "Input must be contiguous FP32 CUDA tensor."
    x = x.contiguous()
    
    batch_size, channels, dim1, dim2, dim3 = x.shape
    
    if stride is None:
        stride = kernel_size
        
    def compute_out_dim(d_in, k, s, p, d, ceil):
        if ceil:
            return int(math.ceil((d_in + 2*p - d*(k-1) - 1) / s + 1))
        else:
            return (d_in + 2*p - d*(k-1) - 1) // s + 1
            
    d1_out = compute_out_dim(dim1, kernel_size, stride, padding, dilation, ceil_mode)
    d2_out = compute_out_dim(dim2, kernel_size, stride, padding, dilation, ceil_mode)
    d3_out = compute_out_dim(dim3, kernel_size, stride, padding, dilation, ceil_mode)
    
    out = torch.empty((batch_size, channels, d1_out, d2_out, d3_out), device=x.device, dtype=torch.float32)
    
    stride_n = channels * dim1 * dim2 * dim3
    stride_c = dim1 * dim2 * dim3
    stride_d1 = dim2 * dim3
    stride_d2 = dim3
    
    num_outputs = batch_size * channels * d1_out * d2_out * d3_out
    grid = (num_outputs,)
    
    maxpool3d_kernel[grid](
        x, out, batch_size, channels, dim1, dim2, dim3,
        d1_out, d2_out, d3_out, kernel_size, stride, padding, dilation,
        stride_n, stride_c, stride_d1, stride_d2
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        
    def forward(self, x):
        return triton_maxpool3d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)