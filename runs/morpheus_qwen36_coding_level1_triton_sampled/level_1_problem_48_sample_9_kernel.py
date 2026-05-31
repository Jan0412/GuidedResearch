import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to compute mean over the last dimension (size K).
    Each program handles one row (size K) and computes the mean.
    """
    pid = tl.program_id(0)
    row_start = pid * K
    
    # Calculate number of blocks needed to cover dimension K
    num_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    row_sum = 0.0
    
    # Loop over blocks to handle K that may not be a multiple of BLOCK_SIZE
    for block in range(num_blocks):
        offsets = row_start + block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_start + K
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        row_sum += tl.sum(x)
        
    # Compute mean and store result
    mean_val = row_sum / K
    tl.store(out_ptr + pid, mean_val)


def triton_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper to launch the Triton mean kernel.
    Assumes x is contiguous and reduction is over the last dimension.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    K = x.shape[-1]
    N = x.numel() // K
    
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128  # Tunable block size
    
    grid = (N,)
    
    mean_kernel[grid](x, out, N, K, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension
    using custom Triton kernels.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean.
        Uses Triton kernel for optimization.
        """
        x_dim = x.dim()
        # Normalize negative dimension
        target_dim = self.dim if self.dim >= 0 else x_dim + self.dim
        
        # If the reduction dimension is already the last dimension, use the kernel directly
        if target_dim == x_dim - 1:
            return triton_mean(x)
        else:
            # Permute the tensor to move the reduction dimension to the last position
            # This allows us to use the optimized last-dimension reduction kernel
            dims = list(range(x_dim))
            dims.remove(target_dim)
            dims.append(target_dim)
            
            x_perm = x.permute(dims).contiguous()
            
            # Compute mean over the last dimension
            out_perm = triton_mean(x_perm)
            
            # The output shape is already correct because permute rearranges dimensions
            # and mean reduces the last one.
            return out_perm


def get_inputs():
    # randomly generate input tensors based on the model architecture
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2)
    return [x]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    dim = 1
    return [dim]