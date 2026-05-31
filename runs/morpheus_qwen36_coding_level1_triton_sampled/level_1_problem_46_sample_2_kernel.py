import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    x_ptr,
    out_ptr,
    N, C, D, H, W,
    k, s, p,
    D_out, H_out, W_out,
    num_w_blocks: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Decode pid to get spatial coordinates and w_block
    w_block = pid % num_w_blocks
    pid //= num_w_blocks
    h_out = pid % H_out
    pid //= H_out
    d_out = pid % D_out
    pid //= D_out
    c = pid % C
    n = pid // C
    
    # w_out offsets for this block
    w_out_offsets = w_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    w_out_mask = w_out_offsets < W_out
    
    # Input window start coordinates
    d_start = d_out * s - p
    h_start = h_out * s - p
    w_start = w_out_offsets * s - p
    
    # Accumulator for sum
    acc = tl.zeros((BLOCK_SIZE_W,), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for d_in in range(k):
        d_idx = d_start + d_in
        d_mask = (d_idx >= 0) & (d_idx < D)
        
        # Skip if d dimension is completely out of bounds
        if not d_mask:
            continue
            
        for h_in in range(k):
            h_idx = h_start + h_in
            h_mask = (h_idx >= 0) & (h_idx < H)
            
            # Skip if h dimension is completely out of bounds
            if not h_mask:
                continue
            
            for w_in in range(k):
                w_idx = w_start + w_in
                # w_idx is a vector of offsets
                w_mask = (w_idx >= 0) & (w_idx < W) & w_out_mask
                
                # Combined mask for loading
                mask = d_mask & h_mask & w_mask
                
                # Compute linear index for x
                # x shape: (N, C, D, H, W)
                # base_idx is scalar for fixed n, c, d, h
                base_idx = ((n * C + c) * D + d_idx) * H * W + h_idx * W
                idx = base_idx + w_idx
                
                # Load values, filling with 0.0 for masked elements
                val = tl.load(x_ptr + idx, mask=mask, other=0.0)
                acc += val
                
    # Average pooling: divide by number of elements in kernel
    acc = acc / (k * k * k)
    
    # Compute output index
    out_idx = ((n * C + c) * D_out + d_out) * H_out * W_out + h_out * W_out + w_out_offsets
    
    # Store result
    tl.store(out_ptr + out_idx, acc, mask=w_out_mask)


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Wrapper function to launch the Triton AvgPool3d kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    N, C, D, H, W = x.shape
    k = kernel_size
    s = stride
    p = padding
    
    # Calculate output dimensions
    D_out = (D + 2 * p - k) // s + 1
    H_out = (H + 2 * p - k) // s + 1
    W_out = (W + 2 * p - k) // s + 1
    
    # Handle case where output dimensions might be <= 0
    if D_out <= 0 or H_out <= 0 or W_out <= 0:
        return torch.empty((N, C, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    out = torch.empty((N, C, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_W = 32
    num_w_blocks = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Total grid size
    grid_size = N * C * D_out * H_out * num_w_blocks
    
    # Launch kernel
    avg_pool3d_kernel[grid_size](
        x, out,
        N, C, D, H, W,
        k, s, p,
        D_out, H_out, W_out,
        num_w_blocks,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the model.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to kernel_size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling using the custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)


def get_inputs():
    """
    Generates random input tensors for the model.
    """
    batch_size = 16
    channels = 32
    depth = 128
    height = 128
    width = 256
    x = torch.rand(batch_size, channels, depth, height, width)
    return [x]


def get_init_inputs():
    """
    Generates initialization inputs for the model.
    """
    kernel_size = 3
    stride = 2
    padding = 1
    return [kernel_size, stride, padding]