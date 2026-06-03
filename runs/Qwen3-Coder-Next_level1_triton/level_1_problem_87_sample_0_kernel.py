import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,  # Input tensor: (B, C_in, H, W)
    w_ptr,  # Weight tensor: (C_out, C_in, 1, 1)
    b_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (B, C_out, H, W)
    B, C_in, C_out, H, W,
    stride_x, stride_cin, stride_h, stride_w,
    stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw,
    stride_out, stride_out_cout, stride_out_h, stride_out_w,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    c_out_block = tl.program_id(1)
    h_block = tl.program_id(2)
    w_block = tl.program_id(3)
    
    # Compute output coordinates
    c_out_start = c_out_block * BLOCK_SIZE_COUT
    h_start = h_block * BLOCK_SIZE_H
    w_start = w_block * BLOCK_SIZE_W
    
    # Create offset ranges
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_COUT)
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks
    c_out_mask = c_out_offsets < C_out
    h_mask = h_offsets < H
    w_mask = w_offsets < W
    
    # Initialize accumulator for output
    output = tl.zeros((BLOCK_SIZE_COUT, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over input channels in blocks
    for c_in_block in range(0, C_in, BLOCK_SIZE_CIN):
        c_in_offsets = c_in_block + tl.arange(0, BLOCK_SIZE_CIN)
        c_in_mask = c_in_offsets < C_in
        
        # Load input block: (C_in_block_size, H_block, W_block)
        # We need to broadcast across batches and channels
        x_offsets = (
            batch_id * stride_x +
            c_in_offsets[:, None, None] * stride_cin +
            h_offsets[None, :, None] * stride_h +
            w_offsets[None, None, :] * stride_w
        )
        
        x = tl.load(
            x_ptr + x_offsets,
            mask=c_in_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :],
            other=0.0
        )
        
        # Load weight block: (C_out_block_size, C_in_block_size, 1, 1)
        w_offsets_cout = c_out_offsets[:, None, None, None]
        w_offsets_cin = c_in_offsets[None, :, None, None]
        w_offsets_kh = tl.arange(0, 1)[None, None, :, None]
        w_offsets_kw = tl.arange(0, 1)[None, None, None, :]
        
        weight_offsets = (
            w_offsets_cout * stride_w_cout +
            w_offsets_cin * stride_w_cin +
            w_offsets_kh * stride_w_kh +
            w_offsets_kw * stride_w_kw
        )
        
        w = tl.load(
            w_ptr + weight_offsets,
            mask=c_out_mask[:, None, None, None] & c_in_mask[None, :, None, None],
            other=0.0
        )
        
        # Perform the pointwise convolution (equivalent to matmul for 1x1 conv)
        # x: [C_in_block, H_block, W_block]
        # w: [C_out_block, C_in_block, 1, 1]
        # Result: [C_out_block, H_block, W_block]
        # We need to sum over C_in dimension
        x_reshaped = x.permute(1, 2, 0)  # [H_block, W_block, C_in_block]
        w_reshaped = w.permute(0, 2, 3, 1)  # [C_out_block, 1, 1, C_in_block]
        
        # Broadcast and multiply, then sum over C_in
        result = tl.sum(w_reshaped * x_reshaped[None, :, :, :], axis=3)
        output += result
        
    # Apply bias if present
    if HAS_BIAS:
        b_offsets = c_out_offsets
        b = tl.load(b_ptr + b_offsets, mask=c_out_mask, other=0.0)
        output += b[:, None, None]
    
    # Store output
    out_offsets = (
        batch_id * stride_out +
        c_out_offsets[:, None, None] * stride_out_cout +
        h_offsets[None, :, None] * stride_out_h +
        w_offsets[None, None, :] * stride_out_w
    )
    
    tl.store(
        out_ptr + out_offsets,
        output.to(x_ptr.dtype),
        mask=c_out_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
    )


class TritonPointwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Extract dimensions
        B, C_in, H, W = x.shape
        C_out, _, _, _ = weight.shape
        
        # Create output tensor
        out = torch.empty((B, C_out, H, W), dtype=x.dtype, device=x.device)
        
        # Compute strides
        stride_x = x.stride(0)
        stride_cin = x.stride(1)
        stride_h = x.stride(2)
        stride_w = x.stride(3)
        
        stride_w_cout = weight.stride(0)
        stride_w_cin = weight.stride(1)
        stride_w_kh = weight.stride(2)
        stride_w_kw = weight.stride(3)
        
        stride_out = out.stride(0)
        stride_out_cout = out.stride(1)
        stride_out_h = out.stride(2)
        stride_out_w = out.stride(3)
        
        # Define block sizes
        BLOCK_SIZE_CIN = 64
        BLOCK_SIZE_COUT = 32
        BLOCK_SIZE_H = 16
        BLOCK_SIZE_W = 16
        
        # Define grid
        grid = (
            B,  # batch dimension
            triton.cdiv(C_out, BLOCK_SIZE_COUT),
            triton.cdiv(H, BLOCK_SIZE_H),
            triton.cdiv(W, BLOCK_SIZE_W)
        )
        
        # Launch kernel
        pointwise_conv2d_kernel[grid](
            x, weight, bias, out,
            B, C_in, C_out, H, W,
            stride_x, stride_cin, stride_h, stride_w,
            stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw,
            stride_out, stride_out_cout, stride_out_h, stride_out_w,
            HAS_BIAS=bias is not None,
            BLOCK_SIZE_CIN=BLOCK_SIZE_CIN,
            BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride_x, ctx.stride_cin, ctx.stride_h, ctx.stride_w = stride_x, stride_cin, stride_h, stride_w
        ctx.stride_w_cout, ctx.stride_w_cin, ctx.stride_w_kh, ctx.stride_w_kw = stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw
        ctx.stride_out, ctx.stride_out_cout, ctx.stride_out_h, ctx.stride_out_w = stride_out, stride_out_cout, stride_out_h, stride_out_w
        ctx.B, ctx.C_in, ctx.C_out, ctx.H, ctx.W = B, C_in, C_out, H, W
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        B, C_in, C_out, H, W = ctx.B, ctx.C_in, ctx.C_out, ctx.H, ctx.W
        
        # Compute gradients
        grad_x = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Compute grad_x = grad_output convolved with flipped weights
            # For 1x1 conv, this is equivalent to matmul
            grad_output_reshaped = grad_output.permute(1, 0, 2, 3).flatten(1)
            weight_reshaped = weight.flatten(1)
            grad_x_reshaped = torch.mm(weight_reshaped, grad_output_reshaped)
            grad_x = grad_x_reshaped.view(C_in, B, H, W).permute(1, 0, 2, 3)
        
        if ctx.needs_input_grad[1]:
            # Compute grad_weight = grad_output * x
            # For 1x1 conv, this is sum over spatial dimensions
            grad_output_reshaped = grad_output.permute(1, 0, 2, 3).flatten(1)
            x_reshaped = x.permute(1, 0, 2, 3).flatten(1)
            grad_weight = torch.mm(grad_output_reshaped, x_reshaped.T).view(C_out, C_in, 1, 1)
        
        if bias is not None and ctx.needs_input_grad[2]:
            # Compute grad_bias = sum over batch and spatial dimensions
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        
        return grad_x, grad_weight, grad_bias


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation using optimized Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        # Keep the same weights but use our custom function
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias_flag = bias
        
        # Initialize weights with same values as original conv1d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Copy weights from original conv1d if it exists, otherwise initialize
        if hasattr(self, 'conv1d'):
            with torch.no_grad():
                self.weight.copy_(self.conv1d.weight)
                if self.bias is not None:
                    self.bias.copy_(self.conv1d.bias)
        else:
            # Default initialization
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return TritonPointwiseConv2d.apply(x, self.weight, self.bias)