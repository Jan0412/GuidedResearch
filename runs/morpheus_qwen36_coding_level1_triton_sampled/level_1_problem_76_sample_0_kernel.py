import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C_in, C_out, L, K, L_out,
    stride_x_b, stride_x_c, stride_x_l,
    stride_w_c, stride_w_c_in, stride_w_k,
    stride_y_b, stride_y_c, stride_y_l,
    stride, dilation,
    BLOCK_SIZE_C_IN: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C_out
    c_out = pid % C_out
    
    # Loop over output length dimension
    for l_out in range(L_out):
        acc = 0.0
        
        # Loop over input channels in blocks
        for c_in_start in range(0, C_in, BLOCK_SIZE_C_IN):
            offsets_c = c_in_start + tl.arange(0, BLOCK_SIZE_C_IN)
            mask_c = offsets_c < C_in
            
            # Loop over kernel size
            for k in range(K):
                # Calculate input spatial index
                l_in = l_out * stride + k * dilation
                
                # Mask for valid input indices
                mask_x = mask_c & (l_in < L)
                
                # Load input and weight blocks
                x_vals = tl.load(
                    x_ptr + b * stride_x_b + offsets_c * stride_x_c + l_in * stride_x_l,
                    mask=mask_x, other=0.0
                )
                w_vals = tl.load(
                    w_ptr + c_out * stride_w_c + offsets_c * stride_w_c_in + k * stride_w_k,
                    mask=mask_c, other=0.0
                )
                
                # Accumulate dot product
                acc += tl.sum(x_vals * w_vals)
        
        # Add bias
        acc += tl.load(b_ptr + c_out, mask=(c_out < C_out), other=0.0)
        
        # Store result
        tl.store(
            y_ptr + b * stride_y_b + c_out * stride_y_c + l_out * stride_y_l,
            acc
        )


def triton_conv1d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, stride: int, dilation: int) -> torch.Tensor:
    """
    Custom Triton implementation of 1D convolution.
    """
    B, C_in, L = x.shape
    C_out, _, K = w.shape
    
    # Calculate output length
    L_out = (L - dilation * (K - 1) - 1) // stride + 1
    
    if L_out < 1:
        return torch.empty(B, C_out, 0, device=x.device, dtype=x.dtype)
    
    y = torch.empty(B, C_out, L_out, device=x.device, dtype=x.dtype)
    
    # Get strides
    stride_x_b, stride_x_c, stride_x_l = x.stride()
    stride_w_c, stride_w_c_in, stride_w_k = w.stride()
    stride_y_b, stride_y_c, stride_y_l = y.stride()
    
    # Grid: one program per (batch, out_channel) pair
    grid = (B * C_out,)
    
    # Block size for input channels (must be power of 2)
    BLOCK_SIZE_C_IN = 64
    
    conv1d_kernel[grid](
        x, w, b, y,
        B, C_in, C_out, L, K, L_out,
        stride_x_b, stride_x_c, stride_x_l,
        stride_w_c, stride_w_c_in, stride_w_k,
        stride_y_b, stride_y_c, stride_y_l,
        stride, dilation,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN
    )
    
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.stride = stride
        self.dilation = dilation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv1d.weight
        b = self.conv1d.bias
        
        # Ensure bias is a tensor for uniform handling
        if b is None:
            b = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
            
        return triton_conv1d(x, w, b, self.stride, self.dilation)