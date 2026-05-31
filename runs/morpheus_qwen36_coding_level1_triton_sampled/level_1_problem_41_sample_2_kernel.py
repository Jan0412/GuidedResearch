import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,
    out_ptr,
    B, F, L_in, L_out, K, D, P,
    BLOCK_SIZE_SEQ: tl.constexpr
):
    """
    Triton kernel for 1D Max Pooling with dilation.
    Each program handles one feature map (batch * feature combination).
    Each thread handles a block of output sequence elements.
    """
    pid = tl.program_id(0)
    x_ptr = x_ptr + pid * L_in
    out_ptr = out_ptr + pid * L_out
    
    thread_idx = tl.program_id(1)
    out_start = thread_idx * BLOCK_SIZE_SEQ
    out_indices = out_start + tl.arange(0, BLOCK_SIZE_SEQ)
    out_mask = out_indices < L_out
    
    # Initialize max values to 0.0 (padding value for MaxPool)
    max_val = tl.zeros((BLOCK_SIZE_SEQ,), dtype=tl.float32)
    
    # Iterate over kernel elements with dilation
    for k in range(K):
        # Compute input indices for this kernel position
        # Virtual input index = out_indices + k * D
        # Actual tensor index = virtual - P
        in_offset = out_indices + k * D - P
        
        # Mask for valid tensor indices
        in_mask = (in_offset >= 0) & (in_offset < L_in)
        
        # Load values, using 0.0 for out-of-bounds (padding)
        val = tl.load(x_ptr + in_offset, mask=in_mask, other=0.0)
        
        # Update max
        max_val = tl.maximum(max_val, val)
        
    # Store results
    tl.store(out_ptr + out_indices, max_val, mask=out_mask)


def triton_maxpool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int
) -> torch.Tensor:
    """
    Wrapper function to launch the Triton MaxPool1d kernel.
    """
    assert x.is_cuda and x.dtype == torch.float32, "Input must be a contiguous FP32 CUDA tensor."
    x = x.contiguous()
    
    B, F, L_in = x.shape
    S = stride
    
    # Calculate output length
    # L_out = floor((L_in + 2*P - D*(K-1) - 1) / S) + 1
    K_eff = 1 + (kernel_size - 1) * dilation
    L_out = (L_in + 2 * padding - K_eff) // S + 1
    
    out = torch.empty((B, F, L_out), dtype=torch.float32, device=x.device)
    
    BLOCK_SIZE_SEQ = 128
    
    # Grid calculation
    num_threads = (L_out + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ
    grid = lambda meta: (B * F, num_threads)
    
    # Launch kernel
    maxpool1d_kernel[grid](
        x, out,
        B, F, L_in, L_out,
        kernel_size, dilation, padding,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        
        if return_indices:
            raise NotImplementedError("return_indices=True is not supported in the Triton implementation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )