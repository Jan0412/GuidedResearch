import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr, 
    out_ptr, 
    B, C, H, W, 
    K, S, P, 
    oH, oW, 
    stride_b, stride_c, stride_h, stride_w, 
    out_stride_b, out_stride_c, out_stride_h, out_stride_w, 
    BLOCK_SIZE_K: tl.constexpr,
):
    # Map program IDs to output coordinates
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Batch and Channel indices
    b = pid_bc // C
    c = pid_bc % C
    
    # Output height and width indices
    oh = pid_h
    ow = pid_w

    # Calculate the top-left corner of the input window
    h_start = oh * S - P
    w_start = ow * S - P

    # Create offsets for the pooling window (must be power of 2)
    kh = tl.arange(0, BLOCK_SIZE_K)
    kw = tl.arange(0, BLOCK_SIZE_K)

    # Broadcast offsets to 2D window
    h_coords = h_start + kh[:, None]
    w_coords = w_start + kw[None, :]

    # Mask for:
    # 1. Window size (K x K)
    # 2. Image boundaries (Padding)
    mask = (kh[:, None] < K) & (kw[None, :] < K) & \
           (h_coords >= 0) & (h_coords < H) & \
           (w_coords >= 0) & (w_coords < W)

    # Compute pointer to the start of the window for the current (b, c)
    base_ptr = x_ptr + b * stride_b + c * stride_c
    # Offset by the 2D coordinates
    ptr = base_ptr + h_coords * stride_h + w_coords * stride_w

    # Load values from the window, using 0.0 for masked elements (padding)
    vals = tl.load(ptr, mask=mask, other=0.0)
    
    # Sum the window elements
    window_sum = tl.sum(vals, axis=0)
    window_sum = tl.sum(window_sum, axis=0)

    # Average pooling: divide by K*K (count_include_pad=True)
    res = window_sum / (K * K)

    # Store the result in the output tensor
    out_ptr_val = out_ptr + b * out_stride_b + c * out_stride_c + oh * out_stride_h + ow * out_stride_w
    tl.store(out_ptr_val, res)

def triton_avg_pool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    # Ensure input is on GPU and contiguous
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    B, C, H, W = x.shape
    S = stride if stride is not None else kernel_size
    P = padding
    K = kernel_size

    # Calculate output dimensions
    oH = (H + 2 * P - K) // S + 1
    oW = (W + 2 * P - K) // S + 1
    
    out = torch.empty((B, C, oH, oW), device=x.device, dtype=x.dtype)
    
    # Tensors strides
    stride_b, stride_c, stride_h, stride_w = x.stride()
    out_stride_b, out_stride_c, out_stride_h, out_stride_w = out.stride()

    # Find the smallest power of 2 >= K for the window block size
    BLOCK_SIZE_K = triton.next_power_of_2(K)

    # Grid: parallelize over (Batch * Channel), Output Height, and Output Width
    grid = (B * C, oH, oW)

    avg_pool_kernel[grid](
        x, out,
        B, C, H, W,
        K, S, P,
        oH, oW,
        stride_b, stride_c, stride_h, stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using custom Triton kernels.
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
        Applies 2D Average Pooling to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)