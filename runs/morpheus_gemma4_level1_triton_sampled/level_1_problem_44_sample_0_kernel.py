import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr, 
    out_ptr,
    stride_xb, stride_xc, stride_xl,
    stride_ob, stride_oc, stride_ol,
    L, L_out, C,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of the output sequence for a specific (batch, channel)
    pid_bc = tl.program_id(0)
    pid_l = tl.program_id(1)

    # Decompose pid_bc into batch and channel indices
    batch_idx = pid_bc // C
    chan_idx = pid_bc % C

    # Base pointers for the current batch and channel
    x_base = x_ptr + batch_idx * stride_xb + chan_idx * stride_xc
    out_base = out_ptr + batch_idx * stride_ob + chan_idx * stride_oc

    # Output sequence offsets for this block
    offsets_l = pid_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_l = offsets_l < L_out

    # Accumulator for the average pooling
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the pooling window
    for k in range(kernel_size):
        # Calculate the input index for the k-th element of the window
        # window_start = offsets_l * stride - padding
        # input_idx = window_start + k
        input_offsets = offsets_l * stride + k - padding
        
        # Mask for input bounds (padding is treated as 0)
        mask_in = (input_offsets >= 0) & (input_offsets < L)
        
        # Load input values and accumulate
        vals = tl.load(x_base + input_offsets * stride_xl, mask=mask_in, other=0.0)
        acc += vals

    # Compute the average
    out = acc / kernel_size
    
    # Store the result
    tl.store(out_base + offsets_l * stride_ol, out, mask=mask_l)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Triton implementation of 1D Average Pooling.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Input shape: (B, C, L)
    B, C, L = x.shape
    
    # Calculate output length
    L_out = (L + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)
    
    # Strides for input and output
    stride_xb, stride_xc, stride_xl = x.stride()
    stride_ob, stride_oc, stride_ol = out.stride()

    # Grid: (B * C, ceil(L_out / BLOCK_SIZE))
    BLOCK_SIZE = 128
    grid = (B * C, triton.cdiv(L_out, BLOCK_SIZE))

    # Launch the Triton kernel
    avg_pool_kernel[grid](
        x, out,
        stride_xb, stride_xc, stride_xl,
        stride_ob, stride_oc, stride_ol,
        L, L_out, C,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied.
        """
        # Ensure input is contiguous for correct stride arithmetic in the kernel
        x = x.contiguous()
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)