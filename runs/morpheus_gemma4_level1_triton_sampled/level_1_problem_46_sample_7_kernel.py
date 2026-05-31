import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool3d_kernel(
    x_ptr, 
    out_ptr, 
    batch_size, 
    channels, 
    in_d, in_h, in_w, 
    out_d, out_h, out_w, 
    stride, 
    padding, 
    K: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
):
    # Grid is (batch_size * channels, out_d, (out_h * out_w + BLOCK_SIZE - 1) // BLOCK_SIZE)
    pid_nc = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Decode batch and channel
    n = pid_nc // channels
    c = pid_nc % channels

    # Decode height and width using block
    hw_offsets = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_hw = hw_offsets < (out_h * out_w)
    
    h = hw_offsets // out_w
    w = hw_offsets % out_w

    # Strides for input tensor (N, C, D, H, W)
    s_n = channels * in_d * in_h * in_w
    s_c = in_d * in_h * in_w
    s_d = in_h * in_w
    s_h = in_w
    s_w = 1

    # Strides for output tensor (N, C, D_out, H_out, W_out)
    os_n = channels * out_d * out_h * out_w
    os_c = out_d * out_h * out_w
    os_d = out_h * out_w
    os_h = out_w
    os_w = 1

    # Starting input coordinates
    in_d_start = pid_d * stride - padding
    in_h_start = h * stride - padding
    in_w_start = w * stride - padding

    # Summing over the kernel window
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(K):
        curr_d = in_d_start + i
        if curr_d >= 0 and curr_d < in_d:
            for j in range(K):
                curr_h = in_h_start + j
                # Mask for height boundary
                mask_h = (curr_h >= 0) & (curr_h < in_h)
                for k in range(K):
                    curr_w = in_w_start + k
                    # Mask for width boundary
                    mask_w = (curr_w >= 0) & (curr_w < in_w)
                    
                    # Combined mask for this element
                    mask = mask_hw & mask_h & mask_w
                    
                    # Input pointer calculation
                    # Note: n, c, pid_d are scalars, h, w are tensors
                    ptr = x_ptr + n * s_n + c * s_c + curr_d * s_d + (curr_h * s_h if mask_h.any() else 0) + (curr_w * s_w if mask_w.any() else 0)
                    # Because h and w are tensors, we must handle the offset correctly
                    # Correcting pointer logic for tensor offsets:
                    ptr = x_ptr + n * s_n + c * s_c + curr_d * s_d + (in_h_start + j) * s_h + (in_w_start + k) * s_w
                    # Wait, in_h_start and in_w_start are tensors. Let's rewrite.
                    
    # Let's refine the pointer logic for the loop to be more robust
    # Resetting acc
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(K):
        id_val = in_d_start + i
        if id_val >= 0 and id_val < in_d:
            for j in range(K):
                ih_val = in_h_start + j
                for k in range(K):
                    iw_val = in_w_start + k
                    
                    # Load mask: check bounds for d, h, w
                    # id_val is scalar, ih_val and iw_val are tensors
                    mask = mask_hw & (ih_val >= 0) & (ih_val < in_h) & (iw_val >= 0) & (iw_val < in_w)
                    
                    # Calculate pointer: 
                    # Base + n*s_n + c*s_c + id_val*s_d + ih_val*s_h + iw_val*s_w
                    ptr = x_ptr + n * s_n + c * s_c + id_val * s_d + ih_val * s_h + iw_val * s_w
                    val = tl.load(ptr, mask=mask, other=0.0)
                    acc += val

    # Average pooling: divide by K^3
    res = acc / (K * K * K)
    
    # Store result
    out_offset = n * os_n + c * os_c + pid_d * os_d + h * os_h + w * os_w
    tl.store(out_ptr + out_offset, res, mask=mask_hw)

def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    n, c, d, h, w = x.shape
    
    # Calculate output dimensions
    out_d = (d + 2 * padding - kernel_size) // stride + 1
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty((n, c, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 128
    grid = (n * c, out_d, (out_h * out_w + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    avg_pool3d_kernel[grid](
        x, out, 
        n, c, d, h, w, 
        out_d, out_h, out_w, 
        stride, padding, 
        K=kernel_size, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)