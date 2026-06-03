import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, H_in, W_in)
    w_ptr,  # Weight tensor pointer (C_in, C_out, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,) or None
    out_ptr,  # Output tensor pointer (N, C_out, H_out, W_out)
    N: tl.constexpr,  # Batch size
    C_in: tl.constexpr,  # Input channels
    C_out: tl.constexpr,  # Output channels
    H_in: tl.constexpr,  # Input height
    W_in: tl.constexpr,  # Input width
    K_h: tl.constexpr,  # Kernel height
    K_w: tl.constexpr,  # Kernel width
    H_out: tl.constexpr,  # Output height
    W_out: tl.constexpr,  # Output width
    stride: tl.constexpr,  # Stride
    padding: tl.constexpr,  # Padding
    dilation: tl.constexpr,  # Dilation
    BLOCK_SIZE_C_out: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_C_in: tl.constexpr,  # Block size for input channels
    BLOCK_SIZE_H: tl.constexpr,  # Block size for output height
    BLOCK_SIZE_W: tl.constexpr,  # Block size for output width
):
    # Get program IDs
    batch_id = tl.program_id(0)
    c_out_block = tl.program_id(1)
    h_block = tl.program_id(2)
    w_block = tl.program_id(3)
    
    # Calculate offsets for output channels, height, and width
    c_out_offsets = c_out_block * BLOCK_SIZE_C_out + tl.arange(0, BLOCK_SIZE_C_out)
    h_offsets = h_block * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid indices
    c_out_mask = c_out_offsets < C_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    
    # Initialize accumulator for output
    output = tl.zeros((BLOCK_SIZE_C_out, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in_start in range(0, C_in, BLOCK_SIZE_C_in):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C_in)
        c_in_mask = c_in_offsets < C_in
        
        # Reshape masks for broadcasting
        c_in_mask_2d = c_in_mask[:, None]
        c_in_mask_3d = c_in_mask[:, None, None]
        
        # Load input values: shape (BLOCK_SIZE_C_in, BLOCK_SIZE_H, BLOCK_SIZE_W)
        # For each output position, we need to look up corresponding input positions
        h_out_idx = h_offsets[None, :, None]
        w_out_idx = w_offsets[None, None, :]
        
        # Calculate corresponding input positions
        h_in = h_out_idx * stride - padding + c_in_start * 0  # Placeholder, will be calculated below
        w_in = w_out_idx * stride - padding + c_in_start * 0  # Placeholder
        
        # We need to loop over kernel positions for each input-output relationship
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate input position for this kernel position
                h_in = h_out_idx * stride - padding + kh * dilation
                w_in = w_out_idx * stride - padding + kw * dilation
                
                # Check if input position is valid
                valid_h = (h_in >= 0) & (h_in < H_in)
                valid_w = (w_in >= 0) & (w_in < W_in)
                valid_idx = valid_h & valid_w
                
                # Load input values at valid positions
                h_in_flat = h_in * W_in + w_in
                x_offsets = batch_id * C_in * H_in * W_in + c_in_offsets[:, None, None] * H_in * W_in + h_in_flat
                
                # Reshape mask
                valid_idx_3d = valid_idx[None, :, :]
                
                # Load input
                x_val = tl.load(
                    x_ptr + x_offsets,
                    mask=valid_idx_3d & c_in_mask_3d,
                    other=0.0
                )
                
                # Load weight values: weight shape is (C_in, C_out, K_h, K_w)
                w_offsets_4d = c_in_offsets[:, None, None] * C_out * K_h * K_w + \
                              c_out_offsets[None, :, None] * K_h * K_w + \
                              kh * K_w + kw
                w_val = tl.load(
                    w_ptr + w_offsets_4d,
                    mask=c_in_mask_2d & c_out_mask[None, :, None],
                    other=0.0
                )
                
                # Multiply and accumulate
                # x_val: (C_in, H_out_block, W_out_block)
                # w_val: (C_in, C_out, 1)
                # Need to broadcast and sum over C_in dimension
                x_val_expanded = x_val[:, :, :]
                w_val_expanded = w_val[:, :, :]
                
                # Compute product and accumulate
                product = x_val_expanded * w_val_expanded
                output += tl.sum(product, axis=0)  # Sum over C_in dimension
    
    # Apply bias if provided
    if b_ptr is not None:
        b_val = tl.load(b_ptr + c_out_offsets, mask=c_out_mask)
        output += b_val[None, :, None]  # Broadcast bias
    
    # Store output
    out_offsets = batch_id * C_out * H_out * W_out + \
                 c_out_offsets[:, None, None] * H_out * W_out + \
                 h_offsets[None, :, None] * W_out + \
                 w_offsets[None, None, :]
    
    tl.store(
        out_ptr + out_offsets,
        output.to(x_ptr.dtype.element_ty),
        mask=c_out_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
    )


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Triton implementation of 2D transposed convolution.
    
    Args:
        x: Input tensor of shape (N, C_in, H_in, W_in)
        weight: Weight tensor of shape (C_in, C_out, K_h, K_w)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K_h - 1) + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1
    
    # Prepare output tensor
    out = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_C_out = 16
    BLOCK_SIZE_C_in = 16
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 4
    
    # Calculate grid dimensions
    grid = (N, (C_out + BLOCK_SIZE_C_out - 1) // BLOCK_SIZE_C_out, 
            (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, 
            (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out, H_in, W_in, K_h, K_w, H_out, W_out,
        stride, padding, dilation,
        BLOCK_SIZE_C_out, BLOCK_SIZE_C_in, BLOCK_SIZE_H, BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same convolution layer
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            bias=bias
        )
        
        # Store parameters for use in forward pass
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using our optimized Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in). 
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Extract parameters from the original layer
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias if self.bias_flag else None
        
        # Call our Triton implementation
        return triton_conv_transpose2d(
            x, weight, bias, 
            self.stride, self.padding, self.dilation
        )