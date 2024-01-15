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

from n2d2.cells.nn import Conv, BatchNorm2d
from n2d2.cells.cell import Sequence
from n2d2.activation import Linear
from n2d2.models.mobilenetv1 import MobileNetv1Extractor, MobileNetv1Head
from n2d2.quantizer import SATAct, SATCell


__all__ = ["MobileNetv1SATQuant"]


class MobileNetv1SATQuant(Sequence):
    """
    Quantized MobileNetV1 with the SAT method

    Args:
        - nb_outputs (int): number of outputs at the end of the model
        - alpha (float): width multiplier
        - w_range (int, optional): range of values possible for the weights
        - a_range (int, optional): range of values possible for the activations
        - quant_mode (str, optional): mode of quantization. Defaults to "Symmetric.
        - start_rand_IT (int, optional): number of iterations before starting 
            to reduce the bitwidth of the quantized cells. It is used for training.
        - end_rand_IT (int, optional): number of iterations when the reduction of the 
            bitwidth of the quantized cells has to stop. It is used for training.

    Reference: Towards Efficient Training for Neural Network Quantization (https://arxiv.org/pdf/1912.10207.pdf)

    """
    def __init__(self, nb_outputs=1000, alpha=1.0, w_range=255, a_range=255,
                 quant_mode="Symmetric", start_rand_IT=0, end_rand_IT=0):
        
        self.extractor = MobileNetv1SATQuantExtractor(alpha, w_range, a_range,
                                                      start_rand_IT=start_rand_IT,
                                                      end_rand_IT=end_rand_IT)
        
        self.head = MobileNetv1SATQuantHead(nb_outputs=nb_outputs, 
                                            alpha=alpha, 
                                            quant_mode=quant_mode)

        super().__init__([self.extractor, self.head])


class MobileNetv1SATQuantHead(MobileNetv1Head):
    """
    Head of the quantized MobileNetV1

    Partly composed of a fully-connected layer that has to be 8-bit quantized (range=255). 
    It is quantized with the SAT method.
    """
    def __init__(self, nb_outputs=1000, alpha=1.0, quant_mode="Symmetric"):
        super().__init__(nb_outputs, alpha)
        self.fc.quantizer = SATCell(range=255, apply_scaling=True, apply_quantization=True, quant_mode=quant_mode)


class MobileNetv1SATQuantExtractor(MobileNetv1Extractor):
    """
    Extractor part of the quantized MobileNetV1

    Notes:
    - Each convolutional layer is quantized with the SAT method by range='w_range'. 
    - Each activation after BatchNorm layer is quantized with the SAT method by range='a_range'. 
    - The range parameter corresponds to 2^(bitwidth) - 1. 
    - The alpha parameter in the constructor is the width multiplier of mobilenet, and NOT the same alpha as in the quantizer
    which is the learned clipping threshold
    
    """
    def __init__(self, alpha=1.0,  w_range=255, a_range=255, quant_mode="Symmetric", start_rand_IT=0, end_rand_IT=0):
        super().__init__(alpha, True)
        for scale in self.__iter__():
            for cell in scale:
                if isinstance(cell, BatchNorm2d):
                    # Last layer before classifier uses alpha=10.0
                    if cell.get_name() == "bn9_1x1":
                        if not end_rand_IT == 0:
                            cell.activation = Linear(quantizer=SATAct(range=255, rand_range=a_range, alpha=10.0,
                                                                      start_rand_IT=start_rand_IT,
                                                                      end_rand_IT=end_rand_IT))
                        else:
                            cell.activation = Linear(quantizer=SATAct(range=a_range, alpha=10.0))
                    else:
                        if not end_rand_IT == 0:
                            cell.activation = Linear(quantizer=SATAct(range=255, rand_range=a_range, alpha=8.0,
                                                                      start_rand_IT=start_rand_IT,
                                                                      end_rand_IT=end_rand_IT))
                        else:
                            cell.activation = Linear(quantizer=SATAct(range=a_range, alpha=8.0))

                if isinstance(cell, Conv):
                    # First layer is always standard conv with 8-bit weights
                    if cell.get_name() == "conv1":
                        cell.quantizer = SATCell(range=255, apply_scaling=False,
                                                 apply_quantization=False, quant_mode=quant_mode)
                    else:
                        cell.quantizer = SATCell(range=w_range, apply_scaling=False,
                                                 apply_quantization=False, quant_mode=quant_mode)
