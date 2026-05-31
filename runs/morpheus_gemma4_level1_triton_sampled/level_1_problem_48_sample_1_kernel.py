import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_reduction_kernel(
    x_ptr, 
    out_ptr, 
    n_red, 
    stride_red, 
    n_out, 
    shapes_ptr, 
    strides_ptr, 
    dim, 
    rank, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    if pid >= n_out:
        return

    # Calculate the starting offset for the reduction group
    # The output index 'pid' is the flattened index of the tensor with the reduction dimension removed.
    # We map this back to the original tensor's coordinates to find the starting pointer.
    offset = 0
    temp_pid = pid
    for i in range(rank - 1, -1, -1):
        if i == dim:
            continue
        # Load shape and stride for the current dimension
        s = tl.load(shapes_ptr + i)
        st = tl.load(strides_ptr + i)
        offset += (temp_pid % s) * st
        temp_pid //= s

    # Reduction loop
    acc = 0.0
    for i in range(0, n_red, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_red
        # Load elements along the reduction dimension
        vals = tl.load(x_ptr + offset + offsets * stride_red, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Store the mean (sum / count)
    tl.store(out_ptr + pid, acc / n_red)

def triton_mean(x: torch.Tensor, dim: int):
    """
    Triton implementation of torch.mean along a specific dimension.
    """
    # Ensure input is on GPU and FP32
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.float().contiguous()
    
    shape = x.shape
    rank = x.ndim
    n_red = shape[dim]
    stride_red = x.stride(dim)
    
    # Calculate output shape and total output elements
    out_shape = [s for i, s in enumerate(shape) if i != dim]
    n_out = x.numel() // n_red
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    
    # Pass shapes and strides as tensors to the kernel
    shapes_tensor = torch.tensor(shape, dtype=torch.int32, device=x.device)
    strides_tensor = torch.tensor(x.stride(), dtype=torch.int32, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = (n_out,)
    
    mean_reduction_kernel[grid](
        x, out, 
        n_red, stride_red, n_out, 
        shapes_tensor, strides_tensor, 
        dim, rank, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using a Triton kernel.
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

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        # Use the custom Triton kernel for mean reduction
        return triton_mean(x, self.dim)