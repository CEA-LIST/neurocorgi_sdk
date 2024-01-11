import onnx
import torch
import numpy as np
import torch.nn as nn
import safetensors.torch

__all__ = ["NeuroCorgiNet"]


class Scaling_FixedPoint(nn.Module):
    def __init__(self, 
                 num_features: int, 
                 mode="Symmetric", 
                 fractional_bits=16, 
                 quant_bits=8, 
                 has_clip=True) -> None:
        super().__init__()
        self.mode = mode
        self.quant_bits = quant_bits
        self.fractional_bits = fractional_bits
        self.has_clip = has_clip
        self.scaling = nn.Parameter(torch.zeros(num_features))
        self.clipping = nn.Parameter(torch.zeros(num_features))

        self.half_factor = float(1 << (self.fractional_bits - 1)) if self.fractional_bits > 0 else 0.0
        self.saturation_max = float((1 << self.quant_bits) - 1)

    def forward(self, x):
        self._check_input_dim(x)

        if self.has_clip:
            x = torch.clip(x, torch.zeros_like(x), self.clipping.view(1, -1, 1, 1).expand(x.shape))

        q = torch.round(x) * self.scaling.view(1, -1, 1, 1).expand(x.shape)
        q = (q + self.half_factor).int() >> self.fractional_bits
        y = torch.clip(q, 0.0, self.saturation_max)
        return y
    
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class NeuroCorgiNet(nn.Module):
    def __init__(self, weights:str=""):
        super().__init__()

        # Block Div2
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.scale1 = Scaling_FixedPoint(32, quant_bits=4)
        self.conv1_3x3_dw = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.relu1_3x3_dw = nn.ReLU(inplace=True)
        self.scale1_3x3_dw = Scaling_FixedPoint(32, quant_bits=4)
        self.conv1_1x1 = nn.Conv2d(32, 64, 1)
        self.relu1_1x1 = nn.ReLU(inplace=True)
        self.scale1_1x1 = Scaling_FixedPoint(64, quant_bits=4)

        # Block Div4
        self.conv2_3x3_dw = nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64)
        self.relu2_3x3_dw = nn.ReLU(inplace=True)
        self.scale2_3x3_dw = Scaling_FixedPoint(64, quant_bits=4)
        self.conv2_1x1 = nn.Conv2d(64, 128, 1)
        self.relu2_1x1 = nn.ReLU(inplace=True)
        self.scale2_1x1 = Scaling_FixedPoint(128, quant_bits=4)
        self.conv3_3x3_dw = nn.Conv2d(128, 128, 3, padding=1, groups=128)
        self.relu3_3x3_dw = nn.ReLU(inplace=True)
        self.scale3_3x3_dw = Scaling_FixedPoint(128, quant_bits=4)
        self.conv3_1x1 = nn.Conv2d(128, 128, 1)
        self.relu3_1x1 = nn.ReLU(inplace=True)
        self.scale3_1x1 = Scaling_FixedPoint(128, quant_bits=4)

        # Block Div8
        self.conv4_3x3_dw = nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128)
        self.relu4_3x3_dw = nn.ReLU(inplace=True)
        self.scale4_3x3_dw = Scaling_FixedPoint(128, quant_bits=4)
        self.conv4_1x1 = nn.Conv2d(128, 256, 1)
        self.relu4_1x1 = nn.ReLU(inplace=True)
        self.scale4_1x1 = Scaling_FixedPoint(256, quant_bits=4)
        self.conv5_3x3_dw = nn.Conv2d(256, 256, 3, padding=1, groups=256)
        self.relu5_3x3_dw = nn.ReLU(inplace=True)
        self.scale5_3x3_dw = Scaling_FixedPoint(256, quant_bits=4)
        self.conv5_1x1 = nn.Conv2d(256, 256, 1)
        self.relu5_1x1 = nn.ReLU(inplace=True)
        self.scale5_1x1 = Scaling_FixedPoint(256, quant_bits=4)
        
        # Block Div16
        self.conv6_3x3_dw = nn.Conv2d(256, 256, 3, stride=2, padding=1, groups=256)
        self.relu6_3x3_dw = nn.ReLU(inplace=True)
        self.scale6_3x3_dw = Scaling_FixedPoint(256, quant_bits=4)
        self.conv6_1x1 = nn.Conv2d(256, 512, 1)
        self.relu6_1x1 = nn.ReLU(inplace=True)
        self.scale6_1x1 = Scaling_FixedPoint(512, quant_bits=4)

        self.conv7_1_3x3_dw = nn.Conv2d(512, 512, 3, padding=1, groups=512)
        self.relu7_1_3x3_dw = nn.ReLU(inplace=True)
        self.scale7_1_3x3_dw = Scaling_FixedPoint(512, quant_bits=4)
        self.conv7_1_1x1 = nn.Conv2d(512, 512, 1)
        self.relu7_1_1x1 = nn.ReLU(inplace=True)
        self.scale7_1_1x1 = Scaling_FixedPoint(512, quant_bits=4)

        self.conv7_2_3x3_dw = nn.Conv2d(512, 512, 3, padding=1, groups=512)
        self.relu7_2_3x3_dw = nn.ReLU(inplace=True)
        self.scale7_2_3x3_dw = Scaling_FixedPoint(512, quant_bits=4)
        self.conv7_2_1x1 = nn.Conv2d(512, 512, 1)
        self.relu7_2_1x1 = nn.ReLU(inplace=True)
        self.scale7_2_1x1 = Scaling_FixedPoint(512, quant_bits=4)

        self.conv7_3_3x3_dw = nn.Conv2d(512, 512, 3, padding=1, groups=512)
        self.relu7_3_3x3_dw = nn.ReLU(inplace=True)
        self.scale7_3_3x3_dw = Scaling_FixedPoint(512, quant_bits=4)
        self.conv7_3_1x1 = nn.Conv2d(512, 512, 1)
        self.relu7_3_1x1 = nn.ReLU(inplace=True)
        self.scale7_3_1x1 = Scaling_FixedPoint(512, quant_bits=4)

        self.conv7_4_3x3_dw = nn.Conv2d(512, 512, 3, padding=1, groups=512)
        self.relu7_4_3x3_dw = nn.ReLU(inplace=True)
        self.scale7_4_3x3_dw = Scaling_FixedPoint(512, quant_bits=4)
        self.conv7_4_1x1 = nn.Conv2d(512, 512, 1)
        self.relu7_4_1x1 = nn.ReLU(inplace=True)
        self.scale7_4_1x1 = Scaling_FixedPoint(512, quant_bits=4)

        self.conv7_5_3x3_dw = nn.Conv2d(512, 512, 3, padding=1, groups=512)
        self.relu7_5_3x3_dw = nn.ReLU(inplace=True)
        self.scale7_5_3x3_dw = Scaling_FixedPoint(512, quant_bits=4)
        self.conv7_5_1x1 = nn.Conv2d(512, 512, 1)
        self.relu7_5_1x1 = nn.ReLU(inplace=True)
        self.scale7_5_1x1 = Scaling_FixedPoint(512, quant_bits=4)

        # Block Div32
        self.conv8_3x3_dw = nn.Conv2d(512, 512, 3, stride=2, padding=1, groups=512)
        self.relu8_3x3_dw = nn.ReLU(inplace=True)
        self.scale8_3x3_dw = Scaling_FixedPoint(512, quant_bits=4)
        self.conv8_1x1 = nn.Conv2d(512, 1024, 1)
        self.relu8_1x1 = nn.ReLU(inplace=True)
        self.scale8_1x1 = Scaling_FixedPoint(1024, quant_bits=4)
        self.conv9_3x3_dw = nn.Conv2d(1024, 1024, 3, padding=1, groups=1024)
        self.relu9_3x3_dw = nn.ReLU(inplace=True)
        self.scale9_3x3_dw = Scaling_FixedPoint(1024, quant_bits=4)
        self.conv9_1x1 = nn.Conv2d(1024, 1024, 1)
        self.relu9_1x1 = nn.ReLU(inplace=True)
        self.scale9_1x1 = Scaling_FixedPoint(1024, quant_bits=4)

        # Since the model is fixed on chip, it is not possible to modify its parameters. 
        # To respect this condition, this model cannot be trained.
        super().train(False)

        # Load model
        if weights:
            self.load(weights)

    
    def load(self, file:str):
        if file.endswith(".onnx"):
            self.load_onnx(file)
        elif file.endswith(".safetensors"):
            self.load_safetensors(file)
        else:
            raise ValueError(f'Not supported format for loading parameters for {__class__.__name__}')

    def load_onnx(self, file:str):
        model = onnx.load(file)

        # Load initializers
        for i in model.graph.initializer:

            # Load parameters for convolutions
            if i.name.startswith("conv"):
                values = np.array(onnx.numpy_helper.to_array(i))
                tensor = torch.from_numpy(values)

                if i.name.endswith("_weights"):
                    self._modules[i.name[:-8]].weight = nn.Parameter(tensor)
                elif i.name.endswith("_biases"):
                    self._modules[i.name[:-7]].bias = nn.Parameter(tensor)
                else:
                    raise RuntimeError(f"Cannot load {i.name} initializer.")

            # Load parameters for scaling layers
            if i.name.startswith("scale"):
                values = np.array(onnx.numpy_helper.to_array(i))
                tensor = torch.from_numpy(values)

                if i.name.endswith("_scaling"):
                    self._modules[i.name[:-8]].scaling = nn.Parameter(tensor)
                elif i.name.endswith("_clipping"):
                    self._modules[i.name[:-9]].clipping = nn.Parameter(tensor)
                else:
                    raise RuntimeError(f"Cannot load {i.name} initializer.")
                
    def load_safetensors(self, file:str):
        safetensors.torch.load_model(self, file)

    def train(self, mode:bool = False):
        if mode:
            raise ValueError("Since the model is fixed on chip, NeuroCorgiNet cannot be trained.")

    def forward(self, x):

        # Div2
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.scale1(x)
        x = self.conv1_3x3_dw(x)
        x = self.relu1_3x3_dw(x)
        x = self.scale1_3x3_dw(x)
        x = self.conv1_1x1(x) 
        x = self.relu1_1x1(x)
        div2 = self.scale1_1x1(x)

        # Div4
        x = self.conv2_3x3_dw(div2)
        x = self.relu2_3x3_dw(x)
        x = self.scale2_3x3_dw(x)
        x = self.conv2_1x1(x)
        x = self.relu2_1x1(x)
        x = self.scale2_1x1(x)
        x = self.conv3_3x3_dw(x)
        x = self.relu3_3x3_dw(x)
        x = self.scale3_3x3_dw(x)
        x = self.conv3_1x1(x)
        x = self.relu3_1x1(x)
        div4 = self.scale3_1x1(x)

        # Div8
        x = self.conv4_3x3_dw(div4)
        x = self.relu4_3x3_dw(x)
        x = self.scale4_3x3_dw(x)
        x = self.conv4_1x1(x)
        x = self.relu4_1x1(x)
        x = self.scale4_1x1(x)
        x = self.conv5_3x3_dw(x)
        x = self.relu5_3x3_dw(x)
        x = self.scale5_3x3_dw(x)
        x = self.conv5_1x1(x)
        x = self.relu5_1x1(x)
        div8 = self.scale5_1x1(x)

        # Div16
        x = self.conv6_3x3_dw(div8)
        x = self.relu6_3x3_dw(x)
        x = self.scale6_3x3_dw(x)
        x = self.conv6_1x1(x)
        x = self.relu6_1x1(x)
        x = self.scale6_1x1(x)

        x = self.conv7_1_3x3_dw(x)
        x = self.relu7_1_3x3_dw(x)
        x = self.scale7_1_3x3_dw(x)
        x = self.conv7_1_1x1(x)
        x = self.relu7_1_1x1(x)
        x = self.scale7_1_1x1(x)

        x = self.conv7_2_3x3_dw(x)
        x = self.relu7_2_3x3_dw(x)
        x = self.scale7_2_3x3_dw(x)
        x = self.conv7_2_1x1(x)
        x = self.relu7_2_1x1(x)
        x = self.scale7_2_1x1(x)

        x = self.conv7_3_3x3_dw(x)
        x = self.relu7_3_3x3_dw(x)
        x = self.scale7_3_3x3_dw(x)
        x = self.conv7_3_1x1(x)
        x = self.relu7_3_1x1(x)
        x = self.scale7_3_1x1(x)

        x = self.conv7_4_3x3_dw(x)
        x = self.relu7_4_3x3_dw(x)
        x = self.scale7_4_3x3_dw(x)
        x = self.conv7_4_1x1(x)
        x = self.relu7_4_1x1(x)
        x = self.scale7_4_1x1(x)

        x = self.conv7_5_3x3_dw(x)
        x = self.relu7_5_3x3_dw(x)
        x = self.scale7_5_3x3_dw(x)
        x = self.conv7_5_1x1(x)
        x = self.relu7_5_1x1(x)
        div16 = self.scale7_5_1x1(x)

        # Div32
        x = self.conv8_3x3_dw(div16)
        x = self.relu8_3x3_dw(x)
        x = self.scale8_3x3_dw(x)
        x = self.conv8_1x1(x)
        x = self.relu8_1x1(x)
        x = self.scale8_1x1(x)
        x = self.conv9_3x3_dw(x)
        x = self.relu9_3x3_dw(x)
        x = self.scale9_3x3_dw(x)
        x = self.conv9_1x1(x)
        x = self.relu9_1x1(x)
        div32 = self.scale9_1x1(x)
    
        return div4, div8, div16, div32

