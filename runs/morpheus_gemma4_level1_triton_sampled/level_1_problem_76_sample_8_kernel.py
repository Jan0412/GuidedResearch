import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C_in, L_in, C_out, K, S, D, L_out,
    stride_xb, stride_xc, stride_xl,
    stride_wo, stride_wi, stride_wk,
    stride_ob, stride_oo, stride_ol,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Program IDs
    pid_b_oc = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    # Map pid_b_oc to batch index b and output channel index oc
    b = pid_b_oc // C_out
    oc = pid_b_oc % C_out
    
    # Output length indices for this block
    l_idx = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = l_idx < L_out
    
    # Initialize accumulator for the output values in this block
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)
    
    # Loop over input channels in blocks
    for ic_start in range(0, C_in, BLOCK_C):
        ic_idx = ic_start + tl.arange(0, BLOCK_C)
        mask_c = ic_idx < C_in
        
        # Loop over the kernel size K
        for k in range(K):
            # Load weight: w[oc, ic_idx, k]
            # Weight shape: (C_out, C_in, K)
            w_off = oc * stride_wo + ic_idx * stride_wi + k * stride_wk
            w_val = tl.load(w_ptr + w_off, mask=mask_c, other=0.0) # (BLOCK_C,)
            
            # Load input: x[b, ic_idx, l_idx * S + k * D]
            # Input shape: (B, C_in, L_in)
            # Broadcase ic_idx and l_idx to create a 2D offset matrix (BLOCK_C, BLOCK_L)
            x_off = (b * stride_xb + 
                     ic_idx[:, None] * stride_xc + 
                     (l_idx[None, :] * S + k * D) * stride_xl)
            
            # Mask ensures we don't read out of bounds of the input tensor
            x_mask = mask_c[:, None] & ((l_idx[None, :] * S + k * D) < L_in)
            x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0) # (BLOCK_C, BLOCK_L)
            
            # Multiply weight (BLOCK_C, 1) by input (BLOCK_C, BLOCK_L) and sum over channels
            acc += tl.sum(w_val[:, None] * x_val, axis=0)
    
    # Store the final result in the output tensor
    out_off = b * stride_ob + oc * stride_oo + l_idx * stride_ol
    tl.store(out_ptr + out_off, acc, mask=mask_l)

def triton_conv1d(x, weight, bias, stride, dilation):
    # Input shapes
    B, C_in, L_in = x.shape
    C_out, _, K = weight.shape
    
    # Calculate output length
    L_out = (L_in - (K - 1) * dilation - 1) // stride + 1
    out = torch.empty((B, C_out, L_out), device=x.device, dtype=x.dtype)
    
    # Tensors must be contiguous for the kernel to work with simple strides
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get strides
    stride_xb, stride_xc, stride_xl = x.stride()
    stride_wo, stride_wi, stride_wk = weight.stride()
    stride_ob, stride_oo, stride_ol = out.stride()
    
    # Tuning parameters
    BLOCK_L = 128
    BLOCK_C = 64 # Sufficient for C_in=64, can be adjusted for larger C_in
    
    # Grid: (Batch * Out_Channels, L_out / BLOCK_L)
    grid = (B * C_out, triton.cdiv(L_out, BLOCK_L))
    
    conv1d_kernel[grid](
        x, weight, out,
        B, C_in, L_in, C_out, K, stride, dilation, L_out,
        stride_xb, stride_xc, stride_xl,
        stride_wo, stride_wi, stride_wk,
        stride_ob, stride_oo, stride_ol,
        BLOCK_L=BLOCK_L, BLOCK_C=BLOCK_C
    )
    
    # Handle bias addition in PyTorch for simplicity and efficiency
    if bias is not None:
        out += bias[None, :, None]
        
    return out

class ModelNew(nn.Module):
    """
    Optimized 1D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use nn.Conv1d to manage weights and bias
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.stride = stride
        self.dilation = dilation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using the Triton kernel.
        """
        # Ensure input is on GPU and FP32
        x = x.cuda().float()
        weight = self.conv1d.weight.cuda().float()
        bias = self.conv1d.bias.cuda().float() if self.conv1d.bias is not None else None
        
        return triton_conv1d(x, weight, bias, self.stride, self.dilation)