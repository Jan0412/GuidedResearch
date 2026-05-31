import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size, channels,
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dilation_d, dilation_h, dilation_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output tensor indices
    pid = tl.program_id(0)
    
    # Calculate output indices from the linear ID
    # Layout: [batch, channel, out_d, out_h, out_w]
    total_out_elements = batch_size * channels * out_d * out_h * out_w
    
    # Handle each block of elements
    num_blocks = (total_out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    if pid >= num_blocks:
        return
        
    start_idx = pid * BLOCK_SIZE
    end_idx = tl.minimum(start_idx + BLOCK_SIZE, total_out_elements)
    
    for idx in range(start_idx, end_idx):
        # Decode linear index to 5D indices
        temp = idx
        out_w_idx = temp % out_w
        temp //= out_w
        out_h_idx = temp % out_h
        temp //= out_h
        out_d_idx = temp % out_d
        temp //= out_d
        c_idx = temp % channels
        b_idx = temp // channels
        
        # Calculate input region boundaries
        in_d_start = out_d_idx * stride_d - pad_d
        in_h_start = out_h_idx * stride_h - pad_h
        in_w_start = out_w_idx * stride_w - pad_w
        
        # Initialize max value to -inf
        max_val = -float('inf')
        
        # Iterate over kernel window
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Calculate actual input position with dilation
                    d_pos = in_d_start + kd * dilation_d
                    h_pos = in_h_start + kh * dilation_h
                    w_pos = in_w_start + kw * dilation_w
                    
                    # Check bounds
                    if (0 <= d_pos < in_d and 
                        0 <= h_pos < in_h and 
                        0 <= w_pos < in_w):
                        # Calculate input index
                        input_idx = (b_idx * channels * in_d * in_h * in_w +
                                    c_idx * in_d * in_h * in_w +
                                    d_pos * in_h * in_w +
                                    h_pos * in_w +
                                    w_pos)
                        # Load and update max
                        val = tl.load(x_ptr + input_idx)
                        max_val = tl.maximum(max_val, val)
        
        # Store result
        tl.store(out_ptr + idx, max_val)


def triton_maxpool3d(
    x: torch.Tensor,
    kernel_size: int, stride: int = None,
    padding: int = 0, dilation: int = 1,
    return_indices: bool = False, ceil_mode: bool = False
) -> torch.Tensor:
    """
    Applies 3D max pooling using Triton kernel.
    """
    if return_indices:
        raise NotImplementedError("return_indices=True is not supported in this Triton implementation")
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Extract input dimensions
    batch_size, channels, in_d, in_h, in_w = x.shape
    
    # Handle stride defaults
    if stride is None:
        stride = kernel_size
    
    # Calculate output dimensions
    if ceil_mode:
        out_d = int((in_d + 2 * padding - dilation * (kernel_size - 1) + stride - 1) // stride + 1)
        out_h = int((in_h + 2 * padding - dilation * (kernel_size - 1) + stride - 1) // stride + 1)
        out_w = int((in_w + 2 * padding - dilation * (kernel_size - 1) + stride - 1) // stride + 1)
    else:
        out_d = int((in_d + 2 * padding - dilation * (kernel_size - 1)) // stride + 1)
        out_h = int((in_h + 2 * padding - dilation * (kernel_size - 1)) // stride + 1)
        out_w = int((in_w + 2 * padding - dilation * (kernel_size - 1)) // stride + 1)
    
    # Create output tensor
    out = torch.empty(batch_size, channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    total_out_elements = batch_size * channels * out_d * out_h * out_w
    grid = lambda meta: ((total_out_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    maxpool3d_kernel[grid](
        x, out,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, kernel_size, kernel_size,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model class using Triton kernel for Max Pooling 3D.
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
            return_indices (bool, optional): Whether to return indices (not supported). Defaults to False.
            ceil_mode (bool, optional): When True, uses ceil for output size calculation. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
        if return_indices:
            raise ValueError("return_indices=True is not supported in this Triton implementation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Max Pooling 3D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return triton_maxpool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=self.return_indices,
            ceil_mode=self.ceil_mode
        )