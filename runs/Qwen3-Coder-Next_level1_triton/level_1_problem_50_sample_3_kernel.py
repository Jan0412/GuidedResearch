import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for Conv2d with kernel_size=11, stride=4, padding=2
@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    # Strides
    input_b_stride, input_c_stride, input_h_stride, input_w_stride,
    weight_oc_stride, weight_ic_stride, weight_kh_stride, weight_kw_stride,
    output_b_stride, output_oc_stride, output_h_stride, output_w_stride,
    # Constants
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute output position
    out_h_idx = pid_h
    out_w_idx = pid_w
    
    # Compute input position (top-left corner of the kernel)
    in_h_start = out_h_idx * STRIDE - PADDING
    in_w_start = out_w_idx * STRIDE - PADDING
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_OC,), dtype=tl.float32)
    
    # Iterate over input channels
    for ic in range(in_channels):
        # Iterate over kernel height
        for kh in range(KERNEL_SIZE):
            # Compute input h position
            in_h_pos = in_h_start + kh
            # Check if within bounds
            if in_h_pos >= 0 and in_h_pos < in_h:
                # Iterate over kernel width
                for kw in range(KERNEL_SIZE):
                    # Compute input w position
                    in_w_pos = in_w_start + kw
                    # Check if within bounds
                    if in_w_pos >= 0 and in_w_pos < in_w:
                        # Compute offsets for input
                        input_offset = (pid_b * input_b_stride + 
                                       ic * input_c_stride + 
                                       in_h_pos * input_h_stride + 
                                       in_w_pos * input_w_stride)
                        
                        # Compute weight offset
                        weight_offset = (pid_oc * weight_oc_stride + 
                                        ic * weight_ic_stride + 
                                        kh * weight_kh_stride + 
                                        kw * weight_kw_stride)
                        
                        # Load input and weight values
                        input_val = tl.load(input_ptr + input_offset)
                        weight_val = tl.load(weight_ptr + weight_offset)
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Store result
    output_offset = (pid_b * output_b_stride + 
                    pid_oc * output_oc_stride + 
                    out_h_idx * output_h_stride + 
                    out_w_idx * output_w_stride)
    tl.store(output_ptr + output_offset, acc)


class TritonConv2d(nn.Module):
    def __init__(self, conv_layer):
        super().__init__()
        # Store the conv layer parameters
        self.conv_layer = conv_layer
        # Extract parameters
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        # Initialize weights and bias from the original conv layer
        self.weight = conv_layer.weight
        self.bias = conv_layer.bias
    
    def forward(self, x):
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels, _, kh, kw = self.weight.shape
        assert kh == self.kernel_size[0] and kw == self.kernel_size[1], "Kernel size mismatch"
        
        # Compute output dimensions
        out_h = (in_h + 2 * self.padding[0] - kh) // self.stride[0] + 1
        out_w = (in_w + 2 * self.padding[1] - kw) // self.stride[1] + 1
        
        # Prepare output tensor
        output = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
        
        # Define block sizes for tiling
        BLOCK_SIZE_B = 1
        BLOCK_SIZE_OC = 32
        BLOCK_SIZE_IC = 3
        BLOCK_SIZE_H = 8
        BLOCK_SIZE_W = 8
        
        # Compute grid dimensions
        grid = (batch_size, 
                triton.cdiv(out_channels, BLOCK_SIZE_OC),
                triton.cdiv(out_h, BLOCK_SIZE_H),
                triton.cdiv(out_w, BLOCK_SIZE_W))
        
        # Compute strides
        input_b_stride = x.stride(0)
        input_c_stride = x.stride(1)
        input_h_stride = x.stride(2)
        input_w_stride = x.stride(3)
        
        weight_oc_stride = self.weight.stride(0)
        weight_ic_stride = self.weight.stride(1)
        weight_kh_stride = self.weight.stride(2)
        weight_kw_stride = self.weight.stride(3)
        
        output_b_stride = output.stride(0)
        output_oc_stride = output.stride(1)
        output_h_stride = output.stride(2)
        output_w_stride = output.stride(3)
        
        # Launch kernel
        conv2d_kernel[grid](
            x,
            self.weight,
            output,
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_channels,
            out_h,
            out_w,
            input_b_stride, input_c_stride, input_h_stride, input_w_stride,
            weight_oc_stride, weight_ic_stride, weight_kh_stride, weight_kw_stride,
            output_b_stride, output_oc_stride, output_h_stride, output_w_stride,
            KERNEL_SIZE=self.kernel_size[0],
            STRIDE=self.stride[0],
            PADDING=self.padding[0],
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_OC=BLOCK_SIZE_OC,
            BLOCK_SIZE_IC=BLOCK_SIZE_IC,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
        )
        
        # Add bias if present
        if self.bias is not None:
            # Bias shape: (out_channels,)
            bias_view = self.bias.view(1, -1, 1, 1)
            output = output + bias_view
        
        return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Create original conv layer
        conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # Wrap it with Triton implementation
        self.conv1 = TritonConv2d(conv1)
    
    def forward(self, x):
        x = self.conv1(x)
        return x