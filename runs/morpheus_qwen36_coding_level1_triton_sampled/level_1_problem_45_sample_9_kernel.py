import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for 2D Average Pooling.
    
    Args:
        x_ptr: Pointer to input tensor (N, C, H, W).
        out_ptr: Pointer to output tensor (N, C, H_out, W_out).
        N: Batch size.
        C: Number of channels.
        H: Input height.
        W: Input width.
        kernel_size: Size of the pooling window (constexpr).
        stride: Stride of the pooling operation (constexpr).
        padding: Padding applied to the input (constexpr).
        BLOCK_SIZE: Number of elements per program block.
    """
    pid = tl.program_id(0)
    
    # Total number of output elements
    # We map each program to a block of output elements
    # Output shape is (N, C, H_out, W_out)
    # We need to compute H_out and W_out
    # H_out = (H + 2*padding - kernel_size) // stride + 1
    # W_out = (W + 2*padding - kernel_size) // stride + 1
    
    # Since H_out and W_out depend on inputs, we compute them here or pass them.
    # For constexpr optimization, we can compute them if they are constant, 
    # but they depend on H and W which are runtime. 
    # However, for a specific model instance, H and W are fixed.
    # To keep the kernel general, we compute H_out and W_out here.
    # Note: Using integer division which matches PyTorch's floor behavior for pooling.
    
    h_out = (H + 2 * padding - kernel_size) // stride + 1
    w_out = (W + 2 * padding - kernel_size) // stride + 1
    
    # Total output elements
    num_out_elements = N * C * h_out * w_out
    
    # Determine the range of output indices this program handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_out_elements
    
    # Decode output index (n, c, h_out, w_out)
    # n = pid // (C * h_out * w_out)
    # rem = pid % (C * h_out * w_out)
    # c = rem // (h_out * w_out)
    # rem = rem % (h_out * w_out)
    # h_out_idx = rem // w_out
    # w_out_idx = rem % w_out
    
    # To avoid large intermediate values and potential overflow, we can compute step by step
    # or use modulo arithmetic carefully.
    # Since N*C*h_out*w_out can be large, we rely on Triton's 64-bit integer support if available,
    # or compute offsets directly.
    
    # Let's compute the base offset for the output pointer
    # out_ptr offset = n * C * H_out * W_out + c * H_out * W_out + h_out_idx * W_out + w_out_idx
    
    # We can compute n, c, h, w for each element in the block
    # offsets are linear indices in the flattened output tensor
    
    # Compute n, c, h, w from linear index 'idx'
    # idx = n * (C * h_out * w_out) + c * (h_out * w_out) + h * w_out + w
    
    # Precompute strides
    stride_c = h_out * w_out
    stride_h = w_out
    stride_n = C * stride_c
    
    # For each element in the block
    # We can unroll or loop if BLOCK_SIZE is large, but Triton handles vectorized loads/stores.
    # However, the decoding depends on the index, so we must compute for each offset.
    
    # To optimize, we can compute the indices for the block.
    # Since BLOCK_SIZE is small (e.g., 128), we can compute indices for each.
    
    # n = offsets // stride_n
    # rem = offsets % stride_n
    # c = rem // stride_c
    # rem = rem % stride_c
    # h = rem // stride_h
    # w = rem % stride_h
    
    # This involves division/modulo which can be expensive.
    # Alternative: Map grid to (n, c, h, w) directly if possible, but 1D grid is easier.
    # Given the constraints, the division approach is acceptable for BLOCK_SIZE=128.
    
    n = offsets // stride_n
    rem = offsets % stride_n
    c = rem // stride_c
    rem = rem % stride_c
    h = rem // stride_h
    w = rem % stride_h
    
    # Calculate input coordinates for the top-left of the pooling window
    # h_in = h * stride - padding
    # w_in = w * stride - padding
    
    h_in = h * stride - padding
    w_in = w * stride - padding
    
    # Accumulator for the sum
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over the kernel window
    # Since kernel_size is constexpr, we can unroll this loop for performance
    for k_h in tl.static_range(kernel_size):
        for k_w in tl.static_range(kernel_size):
            # Absolute input coordinates
            curr_h = h_in + k_h
            curr_w = w_in + k_w
            
            # Check bounds
            # mask_h = curr_h >= 0 & curr_h < H
            # mask_w = curr_w >= 0 & curr_w < W
            # mask_hw = mask_h & mask_w
            
            # Combine masks
            mask_hw = (curr_h >= 0) & (curr_h < H) & (curr_w >= 0) & (curr_w < W)
            
            # Calculate input pointer offset
            # offset_in = n * C * H * W + c * H * W + curr_h * W + curr_w
            
            # Compute strides for input
            stride_c_in = H * W
            stride_h_in = W
            
            offset_in = n * stride_n_in + c * stride_c_in + curr_h * stride_h_in + curr_w
            stride_n_in = C * stride_c_in
            
            # Load value with masking
            val = tl.load(x_ptr + offset_in, mask=mask_hw, other=0.0)
            
            # Accumulate
            acc += val
            
    # Compute average
    # Divisor is kernel_size * kernel_size
    divisor = kernel_size * kernel_size
    out_val = acc / divisor
    
    # Calculate output pointer offset
    # offset_out = n * stride_n + c * stride_c + h * stride_h + w
    offset_out = n * stride_n + c * stride_c + h * stride_h + w
    
    # Store result
    tl.store(out_ptr + offset_out, out_val, mask=mask)


def triton_avg_pool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Wrapper function to launch the Triton average pooling kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    N, C, H, W = x.shape
    
    # Compute output dimensions
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    out = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Block size configuration
    BLOCK_SIZE = 128
    
    # Total number of output elements
    num_out_elements = N * C * H_out * W_out
    
    # Grid configuration
    grid = lambda meta: ((num_out_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    avg_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the model.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling using the custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)