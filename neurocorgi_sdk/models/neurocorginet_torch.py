"""
    (C) Copyright 2023 CEA LIST. All Rights Reserved.
    
    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software. You can use,
    modify and/or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".
    As a counterpart to the access to the source code and rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty and the software's author, the holder of the
    economic rights, and the successive licensors have only limited
    liability.
    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
"""

import torch
import n2d2
import pytorch_to_n2d2
from neurocorgi_sdk.models import NeuroCorgiNet


__all__ = ["NeuroCorgiNet_torch"]
    

class NeuroCorgiNet_torch(torch.nn.Module):
    """Torch wrapper for NeuroCorgiNet 
    
    Expects inputs in [0,1] which are rescaled to [0,255]. 
    This choice has been made because Pytorch preprocessing typically uses the ToTensor() method that 
    scales inputs to [0,1]. This behavior can be disabled with int_input = True

    Args:
        - shape (list): input shape
        - weights_dir (str): path to the parameters folder
        - int_input (bool, optional): boolean to decide if a rescaling from [0,1] to [0,255] is required

    """

    def __init__(self, shape, weights_dir=None, int_input=False):
        super().__init__()

        self.neurocorginet = NeuroCorgiNet([_ for _ in shape], weights_dir)
        self.pytorch_neurocorgi = pytorch_to_n2d2.pytorch_interface.Block(self.neurocorginet)
        self.pytorch_neurocorgi.eval()
        self.int_input = int_input
        self.shape = shape


    def forward(self, x):

        if not self.shape == list(x.shape):
            raise RuntimeError("Expected input shape: " + str(self.shape) + 
                               ", but provided input shape: " + str(x.shape))

        with torch.no_grad():
                    
            # Rescale the inputs from [0,1] to [0,255] if necessary
            if not self.int_input:
                x = x * 255

            conv3_1x1, conv5_1x1, conv7_5_1x1, conv9_1x1, _ = self.pytorch_neurocorgi(x)

            return conv3_1x1, conv5_1x1, conv7_5_1x1, conv9_1x1

    def __str__(self):
        return self.neurocorginet.__str__()

    def __getitem__(self, item):
        n2d2_tensor = self.neurocorginet[0][item].get_outputs().dtoh()
        torch_tensor = pytorch_to_n2d2.pytorch_interface._to_torch(n2d2_tensor.N2D2())
        return torch_tensor
