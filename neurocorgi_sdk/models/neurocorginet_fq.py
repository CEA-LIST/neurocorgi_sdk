# NeuroCorgi SDK, CeCILL-C license

import torch
from torch.autograd.function import Function
import numpy as np
import torch.nn as nn

"""
    Quantization
"""
class q_k(Function):
    """
        This is the quantization module.
        The input and output should be all on the interval [0, 1].
    """
    @staticmethod
    def forward(ctx, input, range):
        
        assert range > 0
        assert torch.all(input >= 0) and torch.all(input <= 1)
        res = torch.round(range * input)
        res.div_(range)
        assert torch.all(res >= 0) and torch.all(res <= 1)
        return res
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class SAT_Quant(nn.Module):
    """Impements SAT operator for activations from
    `"Towards Efficient Training for Neural Network Quantization"
    <https://arxiv.org/abs/1912.10207>`.

    Args:
        alpha (float, optional): quantization parameter
        nb_bits (int, optional): Number of bits used to quantize the activations
        range (int, optional): Number of states used to quantize the activations 
            (if not specified by the user, should be 2**nb_bits - 1) 
    """

    def __init__(self, 
                 alpha:float=1.0, 
                 nb_bits:int=8, 
                 range:int=-1) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.quant = q_k.apply

        tmp_range = range if range != -1 else pow(2, nb_bits) - 1
        self.range = nn.Parameter(torch.tensor([float(tmp_range)]))

    def forward(self, x):
        x_clip = torch.clip(x, 0.0, self.alpha.item())
        q = x_clip / self.alpha
        q = self.quant(q, self.range)
        y = q * self.alpha
        return y


class NeuroCorgiNet_fakequant(nn.Module):
    """NeuroCorgi model on chip

    Simulate the 4-bit fakequant quantized MobileNetV1.\n 
    The paramaters (weights & biases) of the convolutions are quantized in 4-bit but are still stored in floating point 32-bit.
    The batchnormalization layers are not fused with the convolutions. Ideally, use inputs with values between 0 and 1.

    Since the model is fixed on chip, it is not possible to modify its parameters. 
    To respect this condition, this model base mode is eval.
    Training is possible for exploration purpose.

    Use the model::

        >>> model = NeuroCorgiNet_fakequant("model_fakequant.safetensors")
        >>> div4, div8, div16, div32 = model(image)
    """

    def __init__(self, weights:str=""):
        super().__init__()

        # Block Div2
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.satq1 = SAT_Quant(nb_bits=4)
        self.conv1_3x3_dw = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.bn1_3x3_dw = nn.BatchNorm2d(32)
        self.satq1_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv1_1x1 = nn.Conv2d(32, 64, 1, bias=False)
        self.bn1_1x1 = nn.BatchNorm2d(64)
        self.satq1_1x1 = SAT_Quant(nb_bits=4)

        # Block Div4
        self.conv2_3x3_dw = nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64, bias=False)
        self.bn2_3x3_dw = nn.BatchNorm2d(64)
        self.satq2_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv2_1x1 = nn.Conv2d(64, 128, 1, bias=False)
        self.bn2_1x1 = nn.BatchNorm2d(128)
        self.satq2_1x1 = SAT_Quant(nb_bits=4)
        self.conv3_3x3_dw = nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False)
        self.bn3_3x3_dw = nn.BatchNorm2d(128)
        self.satq3_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv3_1x1 = nn.Conv2d(128, 128, 1, bias=False)
        self.bn3_1x1 = nn.BatchNorm2d(128)
        self.satq3_1x1 = SAT_Quant(nb_bits=4)

        # Block Div8
        self.conv4_3x3_dw = nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128, bias=False)
        self.bn4_3x3_dw = nn.BatchNorm2d(128)
        self.satq4_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv4_1x1 = nn.Conv2d(128, 256, 1, bias=False)
        self.bn4_1x1 = nn.BatchNorm2d(256)
        self.satq4_1x1 = SAT_Quant(nb_bits=4)
        self.conv5_3x3_dw = nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False)
        self.bn5_3x3_dw = nn.BatchNorm2d(256)
        self.satq5_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv5_1x1 = nn.Conv2d(256, 256, 1, bias=False)
        self.bn5_1x1 = nn.BatchNorm2d(256)
        self.satq5_1x1 = SAT_Quant(nb_bits=4)
        
        # Block Div16
        self.conv6_3x3_dw = nn.Conv2d(256, 256, 3, stride=2, padding=1, groups=256, bias=False)
        self.bn6_3x3_dw = nn.BatchNorm2d(256)
        self.satq6_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv6_1x1 = nn.Conv2d(256, 512, 1, bias=False)
        self.bn6_1x1 = nn.BatchNorm2d(512)
        self.satq6_1x1 = SAT_Quant(nb_bits=4)

        self.conv7_1_3x3_dw = nn.Conv2d(512, 512, 3, padding=1, groups=512, bias=False)
        self.bn7_1_3x3_dw = nn.BatchNorm2d(512)
        self.satq7_1_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv7_1_1x1 = nn.Conv2d(512, 512, 1, bias=False)
        self.bn7_1_1x1 = nn.BatchNorm2d(512)
        self.satq7_1_1x1 = SAT_Quant(nb_bits=4)

        self.conv7_2_3x3_dw = nn.Conv2d(512, 512, 3, padding=1, groups=512, bias=False)
        self.bn7_2_3x3_dw = nn.BatchNorm2d(512)
        self.satq7_2_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv7_2_1x1 = nn.Conv2d(512, 512, 1, bias=False)
        self.bn7_2_1x1 = nn.BatchNorm2d(512)
        self.satq7_2_1x1 = SAT_Quant(nb_bits=4)

        self.conv7_3_3x3_dw = nn.Conv2d(512, 512, 3, padding=1, groups=512, bias=False)
        self.bn7_3_3x3_dw = nn.BatchNorm2d(512)
        self.satq7_3_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv7_3_1x1 = nn.Conv2d(512, 512, 1, bias=False)
        self.bn7_3_1x1 = nn.BatchNorm2d(512)
        self.satq7_3_1x1 = SAT_Quant(nb_bits=4)

        self.conv7_4_3x3_dw = nn.Conv2d(512, 512, 3, padding=1, groups=512, bias=False)
        self.bn7_4_3x3_dw = nn.BatchNorm2d(512)
        self.satq7_4_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv7_4_1x1 = nn.Conv2d(512, 512, 1, bias=False)
        self.bn7_4_1x1 = nn.BatchNorm2d(512)
        self.satq7_4_1x1 = SAT_Quant(nb_bits=4)

        self.conv7_5_3x3_dw = nn.Conv2d(512, 512, 3, padding=1, groups=512, bias=False)
        self.bn7_5_3x3_dw = nn.BatchNorm2d(512)
        self.satq7_5_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv7_5_1x1 = nn.Conv2d(512, 512, 1, bias=False)
        self.bn7_5_1x1 = nn.BatchNorm2d(512)
        self.satq7_5_1x1 = SAT_Quant(nb_bits=4)

        # Block Div32
        self.conv8_3x3_dw = nn.Conv2d(512, 512, 3, stride=2, padding=1, groups=512, bias=False)
        self.bn8_3x3_dw = nn.BatchNorm2d(512)
        self.satq8_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv8_1x1 = nn.Conv2d(512, 1024, 1, bias=False)
        self.bn8_1x1 = nn.BatchNorm2d(1024)
        self.satq8_1x1 = SAT_Quant(nb_bits=4)
        self.conv9_3x3_dw = nn.Conv2d(1024, 1024, 3, padding=1, groups=1024, bias=False)
        self.bn9_3x3_dw = nn.BatchNorm2d(1024)
        self.satq9_3x3_dw = SAT_Quant(nb_bits=4)
        self.conv9_1x1 = nn.Conv2d(1024, 1024, 1, bias=False)
        self.bn9_1x1 = nn.BatchNorm2d(1024)
        self.satq9_1x1 = SAT_Quant(nb_bits=4)

        # Load model
        if weights:
            self.load(weights)

        # The model is fixed on chip. To respect this, the basic mode is evaluation.
        self.eval()
    
    def load(self, file:str):
        if file.endswith(".onnx"):
            self.load_onnx(file)
        elif file.endswith(".safetensors"):
            self.load_safetensors(file)
        else:
            raise ValueError(f'Not supported format for loading parameters for {__class__.__name__}')

    def load_onnx(self, file:str):
        import onnx
        model = onnx.load(file)

        # Load attributes
        for n in model.graph.node:
            if n.op_type == "SAT_Quant":
                for a in n.attribute:
                    if a.name == "alpha":
                        value = onnx.helper.get_attribute_value(a)
                        self._modules[n.name].alpha = nn.Parameter(torch.tensor([value]))
            
            if n.op_type == "BatchNormalization":
                for a in n.attribute:
                    if a.name == "epsilon":
                        self._modules[n.name].eps = onnx.helper.get_attribute_value(a)
                    if a.name == "momentum":
                        self._modules[n.name].momentum = onnx.helper.get_attribute_value(a)
            
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

            # Load parameters for batchnorms
            if i.name.startswith("bn"):
                values = np.array(onnx.numpy_helper.to_array(i))
                tensor = torch.from_numpy(values)

                if i.name.endswith("_scales"):
                    self._modules[i.name[:-7]].weight = nn.Parameter(tensor)
                elif i.name.endswith("_biases"):
                    self._modules[i.name[:-7]].bias = nn.Parameter(tensor)
                elif i.name.endswith("_means"):
                    self._modules[i.name[:-6]].running_mean = tensor
                elif i.name.endswith("_variances"):
                    self._modules[i.name[:-10]].running_var = tensor
                else:
                    raise RuntimeError(f"Cannot load {i.name} initializer.")

    def load_safetensors(self, file:str):
        import safetensors.torch
        safetensors.torch.load_model(self, file)

    def forward(self, x):

        # Div2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.satq1(x)
        x = self.conv1_3x3_dw(x)
        x = self.bn1_3x3_dw(x)
        x = self.satq1_3x3_dw(x)
        x = self.conv1_1x1(x) 
        x = self.bn1_1x1(x)
        div2 = self.satq1_1x1(x)

        # Div4
        x = self.conv2_3x3_dw(div2)
        x = self.bn2_3x3_dw(x)
        x = self.satq2_3x3_dw(x)
        x = self.conv2_1x1(x)
        x = self.bn2_1x1(x)
        x = self.satq2_1x1(x)
        x = self.conv3_3x3_dw(x)
        x = self.bn3_3x3_dw(x)
        x = self.satq3_3x3_dw(x)
        x = self.conv3_1x1(x)
        x = self.bn3_1x1(x)
        div4 = self.satq3_1x1(x)

        # Div8
        x = self.conv4_3x3_dw(div4)
        x = self.bn4_3x3_dw(x)
        x = self.satq4_3x3_dw(x)
        x = self.conv4_1x1(x)
        x = self.bn4_1x1(x)
        x = self.satq4_1x1(x)
        x = self.conv5_3x3_dw(x)
        x = self.bn5_3x3_dw(x)
        x = self.satq5_3x3_dw(x)
        x = self.conv5_1x1(x)
        x = self.bn5_1x1(x)
        div8 = self.satq5_1x1(x)

        # Div16
        x = self.conv6_3x3_dw(div8)
        x = self.bn6_3x3_dw(x)
        x = self.satq6_3x3_dw(x)
        x = self.conv6_1x1(x)
        x = self.bn6_1x1(x)
        x = self.satq6_1x1(x)

        x = self.conv7_1_3x3_dw(x)
        x = self.bn7_1_3x3_dw(x)
        x = self.satq7_1_3x3_dw(x)
        x = self.conv7_1_1x1(x)
        x = self.bn7_1_1x1(x)
        x = self.satq7_1_1x1(x)

        x = self.conv7_2_3x3_dw(x)
        x = self.bn7_2_3x3_dw(x)
        x = self.satq7_2_3x3_dw(x)
        x = self.conv7_2_1x1(x)
        x = self.bn7_2_1x1(x)
        x = self.satq7_2_1x1(x)

        x = self.conv7_3_3x3_dw(x)
        x = self.bn7_3_3x3_dw(x)
        x = self.satq7_3_3x3_dw(x)
        x = self.conv7_3_1x1(x)
        x = self.bn7_3_1x1(x)
        x = self.satq7_3_1x1(x)

        x = self.conv7_4_3x3_dw(x)
        x = self.bn7_4_3x3_dw(x)
        x = self.satq7_4_3x3_dw(x)
        x = self.conv7_4_1x1(x)
        x = self.bn7_4_1x1(x)
        x = self.satq7_4_1x1(x)

        x = self.conv7_5_3x3_dw(x)
        x = self.bn7_5_3x3_dw(x)
        x = self.satq7_5_3x3_dw(x)
        x = self.conv7_5_1x1(x)
        x = self.bn7_5_1x1(x)
        div16 = self.satq7_5_1x1(x)

        # Div32
        x = self.conv8_3x3_dw(div16)
        x = self.bn8_3x3_dw(x)
        x = self.satq8_3x3_dw(x)
        x = self.conv8_1x1(x)
        x = self.bn8_1x1(x)
        x = self.satq8_1x1(x)
        x = self.conv9_3x3_dw(x)
        x = self.bn9_3x3_dw(x)
        x = self.satq9_3x3_dw(x)
        x = self.conv9_1x1(x)
        x = self.bn9_1x1(x)
        div32 = self.satq9_1x1(x)
    
        return div4, div8, div16, div32
