import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size, channels, dim1, dim2, dim3,
    out_dim1, out_dim2, out_dim3,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output indices
    out_idx = tl.program_id(0)
    batch = out_idx // (channels * out_dim1 * out_dim2 * out_dim3)
    rest = out_idx % (channels * out_dim1 * out_dim2 * out_dim3)
    channel = rest // (out_dim1 * out_dim2 * out_dim3)
    rest = rest % (out_dim1 * out_dim2 * out_dim3)
    d1 = rest // (out_dim2 * out_dim3)
    rest = rest % (out_dim2 * out_dim3)
    d2 = rest // out_dim3
    d3 = rest % out_dim3

    # Calculate input window boundaries
    i1_start = d1 * stride - padding
    i2_start = d2 * stride - padding
    i3_start = d3 * stride - padding

    # Max value initialization
    max_val = -float('inf')

    # Iterate over the pooling window
    for k1 in range(kernel_size):
        i1 = i1_start + k1 * dilation
        for k2 in range(kernel_size):
            i2 = i2_start + k2 * dilation
                for k3 in range(kernel_size):
                    i3 = i3_start + k3 * dilation
                    
                    # Check if within bounds
                    if (i1 >= 0 and i1 < dim1 and 
                        i2 >= 0 and i2 < dim2 and 
                        i3 >= 0 and i3 < dim3):
                        # Calculate input pointer offset
                        offset = (batch * channels * dim1 * dim2 * dim3 +
                                 channel * dim1 * dim2 * dim3 +
                                 i1 * dim2 * dim3 +
                                 i2 * dim3 +
                                 i3)
                        val = tl.load(x_ptr + offset)
                        max_val = tl.maximum(max_val, val)

    # Store result
    tl.store(out_ptr + out_idx, max_val)


def triton_maxpool3d(x, kernel_size, stride, padding, dilation, 
                     out_dim1, out_dim2, out_dim3):
    """Wrapper for maxpool3d Triton kernel"""
    batch_size, channels, dim1, dim2, dim3 = x.shape
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Allocate output tensor
    out = torch.empty(batch_size, channels, out_dim1, out_dim2, out_dim3, 
                     dtype=x.dtype, device=x.device)
    
    # Total number of output elements
    n_elements = batch_size * channels * out_dim1 * out_dim2 * out_dim3
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    maxpool3d_kernel[grid](x, out, batch_size, channels, dim1, dim2, dim3,
                          out_dim1, out_dim2, out_dim3,
                          kernel_size, stride, padding, dilation,
                          BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model with Max Pooling 3D implemented via Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, 
                 dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the optimized Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices. Defaults to False.
            ceil_mode (bool, optional): When True, use ceil instead of floor. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
        # Pre-compute output dimensions
        self._compute_output_dims = True
        
    def _calculate_output_dim(self, input_dim):
        """Calculate output dimension for a given input dimension"""
        if self.ceil_mode:
            return (input_dim + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        else:
            return (input_dim + 2 * self.padding - self.dilation * (self.kernel_size - 1)) // self.stride + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        # Calculate output dimensions
        batch_size, channels, dim1, dim2, dim3 = x.shape
        
        out_dim1 = self._calculate_output_dim(dim1)
        out_dim2 = self._calculate_output_dim(dim2)
        out_dim3 = self._calculate_output_dim(dim3)
        
        return triton_maxpool3d(x, self.kernel_size, self.stride, self.padding, 
                               self.dilation, out_dim1, out_dim2, out_dim3)


# Helper function to compute output dimensions (same as PyTorch)
def _pool_output_size(input_size, kernel_size, stride, padding, dilation, ceil_mode):
    """Compute output size for pooling operation"""
    if ceil_mode:
        return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) // stride + 1
    else:
        return (input_size + 2 * padding - dilation * (kernel_size - 1)) // stride + 1


# Override ModelNew to match PyTorch's exact dimension calculation
class ModelNew(nn.Module):
    """
    Optimized model with Max Pooling 3D implemented via Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, 
                 dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D using Triton kernel.
        """
        batch_size, channels, dim1, dim2, dim3 = x.shape
        
        # Calculate output dimensions using PyTorch's formula
        out_dim1 = (dim1 + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1 + self.stride - 1) // self.stride + 1 if self.ceil_mode else (dim1 + 2 * self.padding - self.dilation * (self.kernel_size - 1)) // self.stride + 1
        out_dim2 = (dim2 + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1 + self.stride - 1) // self.stride + 1 if self.ceil_mode else (dim2 + 2 * self.padding - self.dilation * (self.kernel_size - 1)) // self.stride + 1
        out_dim3 = (dim3 + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1 + self.stride - 1) // self.stride + 1 if self.ceil_mode else (dim3 + 2 * self.padding - self.dilation * (self.kernel_size - 1)) // self.stride + 1
        
        return triton_maxpool3d(x, self.kernel_size, self.stride, self.padding, 
                               self.dilation, out_dim1, out_dim2, out_dim3)