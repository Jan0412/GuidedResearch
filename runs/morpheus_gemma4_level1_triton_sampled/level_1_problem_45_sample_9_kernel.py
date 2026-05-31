import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr, 
    out_ptr,
    h_in, 
    w_in, 
    h_out, 
    w_out,
    stride, 
    padding,
    stride_c, 
    stride_h, 
    stride_w,
    out_stride_c, 
    out_stride_h, 
    out_stride_w,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for 2D Average Pooling.
    Each program computes one output pixel (b, c, oh, ow).
    """
    # Program IDs: pid_bc handles (batch * channels), oh and ow handle output height and width
    pid_bc = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)

    # Calculate the top-left corner of the pooling window in the input tensor
    ih_base = oh * stride - padding
    iw_base = ow * stride - padding

    acc = 0.0
    # Iterate over the kernel window
    for kh in range(0, BLOCK_SIZE_K):
        for kw in range(0, BLOCK_SIZE_K):
            ih = ih_base + kh
            iw = iw_base + kw
            
            # Boundary check for padding: elements outside the input dimensions are treated as 0
            mask = (ih >= 0) & (ih < h_in) & (iw >= 0) & (iw < w_in)
            
            # Calculate the pointer to the input element
            # pid_bc = b * channels + c
            # x_ptr + pid_bc * stride_c + ih * stride_h + iw * stride_w
            ptr = x_ptr + pid_bc * stride_c + ih * stride_h + iw * stride_w
            acc += tl.load(ptr, mask=mask, other=0.0)

    # Calculate pointer to the output element
    out_ptr_val = out_ptr + pid_bc * out_stride_c + oh * out_stride_h + ow * out_stride_w
    
    # Store the average. Default AvgPool2d (count_include_pad=True) divides by kernel_size^2
    tl.store(out_ptr_val, acc / (BLOCK_SIZE_K * BLOCK_SIZE_K))

def triton_avg_pool(x, kernel_size, stride, padding):
    """
    Wrapper function to launch the Triton average pooling kernel.
    """
    # Input dimensions
    batch, channels, h_in, w_in = x.shape
    
    # PyTorch default: stride defaults to kernel_size if not provided
    if stride is None:
        stride = kernel_size
        
    # Calculate output dimensions
    h_out = (h_in + 2 * padding - kernel_size) // stride + 1
    w_out = (w_in + 2 * padding - kernel_size) // stride + 1
    
    # Ensure input is contiguous on GPU
    x = x.contiguous()
    out = torch.empty((batch, channels, h_out, w_out), device=x.device, dtype=x.dtype)
    
    # Strides for input tensor (B, C, H, W)
    # We use pid_bc to index into (B * C)
    stride_c = h_in * w_in
    stride_h = w_in
    stride_w = 1
    
    # Strides for output tensor (B, C, H_out, W_out)
    out_stride_c = h_out * w_out
    out_stride_h = w_out
    out_stride_w = 1
    
    # Grid: (batch * channels, h_out, w_out)
    grid = (batch * channels, h_out, w_out)
    
    # Launch kernel
    avg_pool_kernel[grid](
        x, out,
        h_in, w_in, h_out, w_out,
        stride, padding,
        stride_c, stride_h, stride_w,
        out_stride_c, out_stride_h, out_stride_w,
        BLOCK_SIZE_K=kernel_size
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        # Ensure input is on CUDA
        if not x.is_cuda:
            x = x.cuda()
            
        return triton_avg_pool(x, self.kernel_size, self.stride, self.padding)