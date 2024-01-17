# NeuroCorgi SDK, CeCILL-C license 

import torch
import numpy as np
import torch.nn as nn


class Head4ImageNet(nn.Module):
    def __init__(self, weights:str=""):
        super().__init__()
        self.avgpool = nn.AvgPool2d(7, stride=7)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(1024, 1000)

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
        import onnx
        model = onnx.load(file)

        # Load initializers
        for i in model.graph.initializer:

            # Load parameters for the FC layer
            if i.name.startswith("fc"):
                values = np.array(onnx.numpy_helper.to_array(i))
                tensor = torch.from_numpy(values)

                if i.name.endswith("_weights"):
                    self._modules[i.name[:-8]].weight = nn.Parameter(tensor)
                elif i.name.endswith("_biases"):
                    self._modules[i.name[:-7]].bias = nn.Parameter(tensor)
                else:
                    raise RuntimeError(f"Cannot load {i.name} initializer.")
                
    def load_safetensors(self, file:str):
        import safetensors.torch
        safetensors.torch.load_model(self, file)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x