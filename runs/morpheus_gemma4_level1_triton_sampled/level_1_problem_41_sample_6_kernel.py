import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool1d_kernel(
    x_ptr, 
    out_ptr, 
    L, 
    L_out, 
    stride, 
    padding, 
    dilation, 
    K, 
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for the batch and channel combined, and the output sequence block
    pid_nc = tl.program_id(0)
    pid_l = tl.program_id(1)

    # Calculate the range of output elements this program handles
    offsets_l = pid_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_l = offsets_l < L_out

    # Base pointers for the current (batch, channel) pair
    # x shape is (N, C, L), out shape is (N, C, L_out)
    x_ptr_base = x_ptr + pid_nc * L
    out_ptr_base = out_ptr + pid_nc * L_out

    # Initialize max values to negative infinity
    max_vals = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)

    # Iterate over the kernel window
    for i in range(K):
        # Calculate input offsets for the current kernel element
        # input_idx = output_idx * stride - padding + i * dilation
        input_offsets = offsets_l * stride - padding + i * dilation
        
        # Mask to ensure we are within the valid input sequence boundaries [0, L-1]
        mask_in = (input_offsets >= 0) & (input_offsets < L)
        
        # Load values from input; use -inf for padded regions
        vals = tl.load(x_ptr_base + input_offsets, mask=mask_in, other=-float('inf'))
        
        # Update the running maximum
        max_vals = tl.maximum(max_vals, vals)

    # Store the final maximum values into the output tensor
    tl.store(out_ptr_base + offsets_l, max_vals, mask=mask_l)

def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    # Ensure input is contiguous on GPU
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    N, C, L = x.shape
    
    # Calculate output sequence length
    L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((N, C, L_out), device=x.device, dtype=x.dtype)
    
    # Triton kernel parameters
    BLOCK_SIZE = 1024
    # Grid: (N * C) programs for the batch/channel dimension, 
    # and ceil(L_out / BLOCK_SIZE) programs for the sequence dimension.
    grid = (N * C, (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch the kernel
    maxpool1d_kernel[grid](
        x, out, 
        L, L_out, 
        stride, padding, dilation, 
        kernel_size, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        
        # Keep the original for fallback if return_indices is True, as Triton kernel here only computes values
        if self.return_indices:
            self.maxpool = nn.MaxPool1d(
                kernel_size=kernel_size, stride=self.stride, padding=padding, 
                dilation=dilation, return_indices=return_indices
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.
        """
        if self.return_indices:
            # Fallback to PyTorch if indices are required
            return self.maxpool(x)
        
        # Use optimized Triton kernel for value computation
        return triton_maxpool1d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )