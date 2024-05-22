import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from enum import Enum
import warnings
sys.path.append("KAN_Implementations/Efficient_KAN")
sys.path.append("KAN_Implementations/Original_KAN")
sys.path.append("KAN_Implementations/Fast_KAN")
from efficient_kan import Efficient_KANLinear
from original_kan import KAN
from fast_kan import Fast_KANLinear

class ConvKAN(nn.Module):
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride=1, 
                padding=0, 
                grid_size=5,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                enable_standalone_scale_spline=True,
                base_activation=torch.nn.SiLU(),
                grid_eps=0.02,
                grid_range=[-1, 1],
                sp_trainable=True, 
                sb_trainable=True,
                bias_trainable=True,
                symbolic_enabled=True,
                device="cpu",
                version= "Efficient",
                ):
        super(ConvKAN, self).__init__()

        self.version = version
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)

        self.linear = None
        
        if self.version == "Efficient":
            self.linear = Efficient_KANLinear(
                in_features = in_channels * kernel_size * kernel_size,
                out_features = out_channels,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                enable_standalone_scale_spline=enable_standalone_scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                ) 
            warnings.warn('Warning: Efficient KAN implementation does not support the following parameters: [sp_trainable, sb_trainable, device]') 
        elif self.version == "Original":
            self.linear = KAN(
               width = [in_channels * kernel_size * kernel_size, out_channels],
               grid = grid_size,
               k = spline_order,
               noise_scale = scale_noise,
               noise_scale_base = scale_base,
                base_fun = base_activation,
                symbolic_enabled=symbolic_enabled,
                bias_trainable = bias_trainable,
                grid_eps = grid_eps,
                grid_range = grid_range,
                sp_trainable = sp_trainable,
                sb_trainable = sb_trainable,
                device = device,
         )
               
        elif self.version == "Fast":
            self.linear = Fast_KANLinear(
                input_dim = in_channels * kernel_size * kernel_size,
                output_dim = out_channels,
                num_grids=grid_size,
                spline_weight_init_scale=scale_spline,
                base_activation=base_activation,
                grid_min = grid_range[0],
                grid_max = grid_range[1],
                ) 
            warnings.warn('Warning: Fast KAN implementation does not support the following parameters: [scale_noise, scale_base, enable_standalone_scale_spline, grid_eps, sp_trainable, sb_trainable, device]') 


            
    def forward(self, x):  

        batch_size, in_channels, height, width = x.size()
        assert x.dim() == 4
        assert in_channels == self.in_channels

        # Unfold the input tensor to extract flattened sliding blocks from a batched input tensor.
        # Input:  [batch_size, in_channels, height, width]
        # Output: [batch_size, in_channels*kernel_size*kernel_size, num_patches]
        patches = self.unfold(x)

        # Transpose to have the patches dimension last.
        # Input:  [batch_size, in_channels*kernel_size*kernel_size, num_patches]
        # Output: [batch_size, num_patches, in_channels*kernel_size*kernel_size]
        patches = patches.transpose(1, 2) 
        
        # Reshape the patches to fit the linear layer input requirements.
        # Input:  [batch_size, num_patches, in_channels*kernel_size*kernel_size]
        # Output: [batch_size*num_patches, in_channels*kernel_size*kernel_size]
        patches = patches.reshape(-1, in_channels * self.kernel_size * self.kernel_size) 
        
        # Apply the linear layer to each patch.
        # Input:  [batch_size*num_patches, in_channels*kernel_size*kernel_size]
        # Output: [batch_size*num_patches, out_channels
        out = self.linear(patches)
        
        # Reshape the output to the normal format
        # Input:  [batch_size*num_patches, out_channels]
        # Output: [batch_size, num_patches, out_channels]
        out = out.view(batch_size, -1, out.size(-1))  

        # Calculate the height and width of the output.
        out_height = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # Transpose back to have the channel dimension in the second position.
        # Input:  [batch_size, num_patches, out_channels]
        # Output: [batch_size, out_channels, num_patches]
        out = out.transpose(1, 2)
        
        # Reshape the output to the final shape 
        # Input:  [batch_size, out_channels, num_patches]
        # Output: [batch_size, out_channels, out_height, out_width]
        out = out.view(batch_size, self.out_channels, out_height, out_width) 
        
        return out
