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

import n2d2
from n2d2.cells.cell import DeepNetCell
from neurocorgi_sdk.models import MobileNetv1SATQuant


class NeuroCorgiNet(DeepNetCell):
    """NeuroCorgi model

    Import 4-bit fake quantized MobileNetV1 and convert it to fixed point network.\n  
    Since the model is fixed on chip, it is not possible to modify its parameters. 
    To respect this condition, this model cannot be trained.

    Args:
        - dims (list): torch format [batch_size, channels, X, Y]. Must be consistent with dims already used.
        - weights_dir (str): path to the parameters folder 
        - mode (str): configuration mode for the model (default="int4"). Available modes: 
            - fakequant (the model is quantized but the parameters are stored in floating-point, the outputs are in floating-point)
            - int4 (the inferences are performed in integers, the outputs are integer)

    """

    def __init__(self, dims, weights_dir=None, mode="int4"):

        model = MobileNetv1SATQuant(alpha=1.0, w_range=15, a_range=15)

        if weights_dir is None:
            raise RuntimeError("NeuroCorgiNet needs to know the path to the folder of the parameters.")
        else:
            model.import_free_parameters(weights_dir, ignore_not_exists=False)

        self.input_dims = dims
        self.mode = mode

        dummy_deepnet = model(n2d2.Tensor(dims, value=0)).get_deepnet()

        # This mode will generate the model topology used on chip
        if mode == "int4":

            # fuse_qat required an existing graph
            # We therefore create a dummy provider and deepnet
            dummy_provider = n2d2.provider.DataProvider(
                database= n2d2.database.Database(),
                size=[dims[3], dims[2], dims[1]], 
                batch_size=dims[0]
            )
            
            n2d2.quantizer.fuse_qat(dummy_deepnet,
                                    dummy_provider,
                                    act_scaling_mode='FIXED_MULT32',
                                    w_mode='RINTF',
                                    b_mode='RINTF',
                                    c_mode='RINTF')
    

        super().__init__(dummy_deepnet.N2D2())

        if mode == "int4":
            self.div4 = self['conv3_1x1'].get_outputs()
            self.div8 = self['conv5_1x1'].get_outputs()
            self.div16 = self['conv7_5_1x1'].get_outputs()
            self.div32 = self['conv9_1x1'].get_outputs()

        elif mode == "fakequant":            
            self.div4 = self['bn3_1x1'].get_outputs()
            self.div8 = self['bn5_1x1'].get_outputs()
            self.div16 = self['bn7_5_1x1'].get_outputs()
            self.div32 = self['bn9_1x1'].get_outputs()

        else:
            raise RuntimeError(f"The mode {self.mode} provided is not supported by this SDK.")

        # Setup the model in test mode (learning is not possible with this model)
        self.test()

    def __call__(self, x):
        if x.shape() != self.input_dims:
            raise RuntimeError("Expected input shape " + str(self.input_dims) +
                               ", but provided input shape: " + str(x.shape()))
        x = super().__call__(x)
        # print(self['conv9_1x1'].activation.scaling.getFixedPointScaling().getScalingPerOutput())
        return self.div4, self.div8, self.div16, self.div32, x
    
    def get_scaling(self, output):
        if self.mode == "int4":
            if output == "div4":
                layer = 'conv3_1x1'
            elif output == "div8":
                layer = 'conv5_1x1'
            elif output == "div16":
                layer = 'conv7_5_1x1'
            elif output == "div32":
                layer = 'conv9_1x1'
        else:
            raise RuntimeError(f"The mode {self.mode} provided cannot provided the scaling for the outputs of the model.")
        
        return self[layer].activation.scaling.getFixedPointScaling().getScalingPerOutput()

    # Override block set_solver to avoid that learning rates are modified
    def set_solver(self):
        raise RuntimeError("NeuroCorgiNet is not trainable")

    def update(self):
        raise RuntimeError("NeuroCorgiNet is not trainable")

    def learn(self):
        raise RuntimeError("NeuroCorgiNet is not trainable")