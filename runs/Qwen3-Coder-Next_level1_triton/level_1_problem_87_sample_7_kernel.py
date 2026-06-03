import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pointwise_conv_kernel(
    x_ptr,          # Input tensor (B, IC, H, W)
    w_ptr,          # Weight tensor (OC, IC, 1, 1)
    b_ptr,          # Bias tensor (OC,) - can be None
    out_ptr,        # Output tensor (B, OC, H, W)
    B, IC, OC, HW,  # Batch size, input channels, output channels, height*width
    stride_xb, stride_xc, stride_xh,  # Strides for input tensor
    stride_woc, stride_wic,           # Strides for weight tensor
    stride_ob, stride_oc, stride_oh,  # Strides for output tensor
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
):
    # Program ID represents a combination of batch, output channel, and spatial block
    pid_b = tl.program_id(0) // OC
    pid_oc = tl.program_id(0) % OC
    
    # Get the spatial block offset
    hw_block_start = tl.program_id(1) * BLOCK_SIZE_HW
    hw_offsets = hw_block_start + tl.arange(0, BLOCK_SIZE_HW)
    hw_mask = hw_offsets < HW
    
    # Calculate base pointers for this batch and output channel
    x_batch_offset = pid_b * stride_xb
    w_out_offset = pid_oc * stride_woc
    
    # Load bias if present
    bias_val = 0.0
    if HAS_BIAS:
        bias_val = tl.load(b_ptr + pid_oc)
    
    # Accumulate over input channels
    acc = tl.zeros((BLOCK_SIZE_HW,), dtype=tl.float32)
    
    for ic in range(0, IC, BLOCK_SIZE_IC):
        ic_offsets = ic + tl.arange(0, BLOCK_SIZE_IC)
        ic_mask = ic_offsets < IC
        
        # Create masks for both dimensions
        ic_mask_2d = ic_mask[None, :] & hw_mask[:, None]
        
        # Load input: (BLOCK_SIZE_HW, BLOCK_SIZE_IC)
        x_ptr_batch = x_ptr + x_batch_offset + ic_offsets[None, :] * stride_xc + hw_offsets[:, None] * stride_xh
        x_block = tl.load(x_ptr_batch, mask=ic_mask_2d, other=0.0)
        
        # Load weights: (BLOCK_SIZE_IC,)
        w_ptr_out = w_ptr + w_out_offset + ic_offsets * stride_wic
        w_block = tl.load(w_ptr_out, mask=ic_mask, other=0.0)
        
        # Accumulate: broadcast weights across spatial dimension
        acc += tl.sum(x_block * w_block[None, :], axis=1)
    
    # Add bias and convert to output type
    out = acc + bias_val
    
    # Store result
    out_ptr_batch = out_ptr + pid_b * stride_ob + pid_oc * stride_oc + hw_offsets * stride_oh
    tl.store(out_ptr_batch, out, mask=hw_mask)


def triton_pointwise_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    Performs pointwise (1x1) convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (B, IC, H, W)
        weight: Weight tensor of shape (OC, IC, 1, 1)
        bias: Optional bias tensor of shape (OC,)
    
    Returns:
        Output tensor of shape (B, OC, H, W)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, IC, H, W = x.shape
    OC, _, _, _ = weight.shape
    HW = H * W
    
    # Prepare output tensor
    out = torch.empty(B, OC, H, W, device=x.device, dtype=x.dtype)
    
    # Calculate strides
    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_woc, stride_wic, _, _ = weight.stride()
    stride_ob, stride_oc, stride_oh, stride_ow = out.stride()
    
    # Determine grid dimensions
    # Each block handles one (batch, output_channel) combination and a range of spatial positions
    num_batches_oc = B * OC
    BLOCK_SIZE_HW = 256
    BLOCK_SIZE_IC = 64  # Tunable parameter
    
    grid = lambda meta: (
        num_batches_oc,
        (HW + meta['BLOCK_SIZE_HW'] - 1) // meta['BLOCK_SIZE_HW']
    )
    
    # Launch kernel
    pointwise_conv_kernel[grid](
        x, weight, bias, out,
        B, IC, OC, HW,
        stride_xb, stride_xc, stride_xh,
        stride_woc, stride_wic,
        stride_ob, stride_oc, stride_oh,
        HAS_BIAS=bias is not None,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        BLOCK_SIZE_IC=BLOCK_SIZE_IC,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized pointwise 2D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        # Extract and store the weight and bias from the original conv layer
        # We'll use these directly in our Triton implementation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Get weight and bias from the internal conv layer
        weight = self.conv1d.weight.squeeze(-1).squeeze(-1)  # Shape: (out_channels, in_channels)
        bias = self.conv1d.bias if self.conv1d.bias is not None else None
        
        # Reshape weight to (OC, IC, 1, 1) for compatibility
        weight_expanded = weight.unsqueeze(-1).unsqueeze(-1)
        
        return triton_pointwise_conv(x, weight_expanded, bias)