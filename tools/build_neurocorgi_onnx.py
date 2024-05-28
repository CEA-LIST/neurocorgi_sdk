"""
This script generates two onnx file for each version of NeuroCorgiNet
* FakeQuant
* Chip

To use it, run
python -m build_onnx
"""

import numpy as np
import onnx
import os
import math
from onnx import helper
from onnx import numpy_helper
from onnx import TensorProto

##########################################################################
########################### Script variables #############################
##########################################################################

producer_name = "n2d2"
producer_version = "1.3.0"
producer_graph = "n2d2_neurocorgi"

fq_filename = "neurocorginet_fq_imagenet.onnx"
chip_filename = "neurocorginet_imagenet.onnx"

alpha_mobilenet = 1.0
out_mv = lambda x: int(x * 32 * alpha_mobilenet)

inputs_dimensions = [1, 3, 224, 224]

weight_directory = "imagenet_weights"


##########################################################################
########################### Utils functions ##############################
##########################################################################

def get_files_starting_with(key, path):
    filenames = []
    for filename in os.listdir(path):
        if filename.startswith(key):
            filenames.append(filename)
    return filenames

def load_from_syntxt(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        values = [float(val) for line in lines for val in line.strip().split(' ')]
    return np.array(values).astype(np.float32)


##########################################################################
############################ Layer classes ###############################
##########################################################################

class Conv:
    def __init__(self, name, nb_channels=0, nb_outputs=0, kernels=[1,1], stride=[1,1], padding=[0,0], grouping=1, dilation=[1,1]) -> None:
        self.name = name
        self.op_type = "Conv"
        self.weights = 0
        self.biases = 0
        self.nb_channels = nb_channels
        self.nb_outputs = nb_outputs
        self.kernels = kernels
        self.stride = stride
        self.padding = padding
        self.grouping = grouping
        self.dilation = dilation

        self.input_dims = [0] * 4
        self.output_dims = [0] * 4

        if len(self.padding) == 2:
            self.padding = self.padding + self.padding

    def compute_dims(self):
        self.output_dims[0] = self.input_dims[0]
        self.output_dims[1] = self.nb_outputs

        out_h = math.floor(1 + ((self.input_dims[2] - self.kernels[0] + 2 * self.padding[0]) / self.stride[0]))
        out_w = math.floor(1 + ((self.input_dims[3] - self.kernels[1] + 2 * self.padding[1]) / self.stride[1]))

        self.output_dims[2] = out_h
        self.output_dims[3] = out_w

    def load_from_n2d2(self, path):
        # Weights
        file = get_files_starting_with(self.name + "_quant_weights", path)
        self.weights = load_from_syntxt(path + "/" + file[0])

        # Biases
        # No biases provided but init them as zeros
        self.biases = np.zeros(self.nb_outputs)
        
        # Regular conv
        if self.grouping == 1:
            self.weights = self.weights.reshape((self.nb_outputs, self.nb_channels, self.kernels[1], self.kernels[0]))
        elif self.grouping == self.nb_outputs:
            self.weights = self.weights.reshape((self.nb_outputs, 1, self.kernels[1], self.kernels[0]))
        else:
            print("No support for grouping != 1 and != nb_outputs")


class BatchNormalization:
    def __init__(self, name) -> None:
        self.name = name
        self.op_type = "BatchNormalization"
        self.epsilon = 0.000009999999747378752
        self.momentum = 0.8999999761581421
        self.biases = 0
        self.means = 0
        self.scales = 0
        self.variances = 0

        self.input_dims = [0] * 4
        self.output_dims = [0] * 4

    def compute_dims(self):
        self.output_dims = self.input_dims

    def load_from_n2d2(self, path):
        # Scale
        file = get_files_starting_with(self.name + "_scales", path)
        self.scales = load_from_syntxt(path + "/" + file[0])
        
        # Mean
        file = get_files_starting_with(self.name + "_means", path)
        self.means = load_from_syntxt(path + "/" + file[0])

        # Var
        file = get_files_starting_with(self.name + "_variances", path)
        self.variances = load_from_syntxt(path + "/" + file[0])

        # Biases
        file = get_files_starting_with(self.name + "_biases", path)
        self.biases = load_from_syntxt(path + "/" + file[0])


class ScalingFloatingPoint:
    def __init__(self, name) -> None:
        self.name = name
        self.op_type = "ScalingFloatingPoint"

        self.input_dims = [0] * 4
        self.output_dims = [0] * 4

    def compute_dims(self):
        self.output_dims = self.input_dims

    def load_from_n2d2(self, path):
        pass


class ScalingFixedPoint:
    def __init__(self, name, mode, fractional_bits, quant_bits) -> None:
        self.name = name
        self.op_type = "ScalingFixedPoint"
        self.mode = mode
        self.fractional_bits = fractional_bits
        self.quant_bits = quant_bits
        self.scaling = []
        self.clipping = []

        self.input_dims = [0] * 4
        self.output_dims = [0] * 4

    def compute_dims(self):
        self.output_dims = self.input_dims

    def load_from_n2d2(self, path):
        pass


class GlobalAveragePool:
    def __init__(self, name) -> None:
        self.name = name
        self.op_type = "GlobalAveragePool"

        self.input_dims = [0] * 4
        self.output_dims = [0] * 4

    def compute_dims(self):
        self.output_dims[0] = self.input_dims[0]
        self.output_dims[1] = self.input_dims[1]
        self.output_dims[2] = 1
        self.output_dims[3] = 1

    def load_from_n2d2(self, path):
        pass


class Relu:
    def __init__(self, name) -> None:
        self.name = name
        self.op_type = "Relu"

        self.input_dims = [0] * 4
        self.output_dims = [0] * 4

    def compute_dims(self):
        self.output_dims = self.input_dims

    def load_from_n2d2(self, path):
        pass


class Flatten:
    def __init__(self, name) -> None:
        self.name = name
        self.op_type = "Flatten"

        self.input_dims = [0] * 4
        self.output_dims = [0] * 2

    def compute_dims(self):
        self.output_dims[0] = self.input_dims[0]
        self.output_dims[1] = self.input_dims[1]

    def load_from_n2d2(self, path):
        pass


class Gemm:
    def __init__(self, name, nb_inputs, nb_outputs) -> None:
        self.name = name
        self.op_type = "Gemm"
        self.alpha = 1.0
        self.beta = 1.0
        self.transA = 0
        self.transB = 1
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.weights = 0
        self.biases = 0

        self.input_dims = [0] * 2
        self.output_dims = [0] * 2

    def compute_dims(self):
        self.output_dims[0] = self.input_dims[0]
        self.output_dims[1] = self.nb_outputs

    def load_from_n2d2(self, path):
        # Quant weights
        file = get_files_starting_with(self.name + "_quant_weights", path)
        self.weights = load_from_syntxt(path + "/" + file[0])
        self.weights = self.weights.reshape((self.nb_outputs, self.nb_inputs))

        # Biases
        file = get_files_starting_with(self.name + "_biases", path)
        self.biases = load_from_syntxt(path + "/" + file[0])


class SAT_Quant:
    def __init__(self, name, alpha=0, nb_bits=8) -> None:
        self.name = name
        self.op_type = "SAT_Quant"
        self.alpha = alpha
        self.nb_bits = nb_bits
        self.range = (1 << nb_bits) - 1

        self.input_dims = [0] * 4
        self.output_dims = [0] * 4

    def compute_dims(self):
        self.output_dims = self.input_dims

    def load_from_n2d2(self, path):
        name = "bn" + self.name[4:] + "_QAct_SAT_Alpha"
        file = get_files_starting_with(name, path)
        self.alpha = load_from_syntxt(path + "/" + file[0])


##########################################################################
########################## Models definition #############################
##########################################################################

FQ_NeurocorgiModel = [

    # Div2
    Conv("conv1", 3, out_mv(1), kernels=[3,3], stride=[2,2], padding=[1,1]),
    BatchNormalization("bn1"),
    SAT_Quant("satq1", nb_bits=4),
    Conv("conv1_3x3_dw", out_mv(1), out_mv(1), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(1)),
    BatchNormalization("bn1_3x3_dw"),
    SAT_Quant("satq1_3x3_dw", nb_bits=4),
    Conv("conv1_1x1", out_mv(1), out_mv(2)),
    BatchNormalization("bn1_1x1"),
    SAT_Quant("satq1_1x1", nb_bits=4),

    # Div4
    Conv("conv2_3x3_dw", out_mv(2), out_mv(2), kernels=[3,3], stride=[2,2], padding=[1,1], grouping=out_mv(2)),
    BatchNormalization("bn2_3x3_dw"),
    SAT_Quant("satq2_3x3_dw", nb_bits=4),
    Conv("conv2_1x1", out_mv(2), out_mv(4)),
    BatchNormalization("bn2_1x1"),
    SAT_Quant("satq2_1x1", nb_bits=4),
    Conv("conv3_3x3_dw", out_mv(4), out_mv(4), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(4)),
    BatchNormalization("bn3_3x3_dw"),
    SAT_Quant("satq3_3x3_dw", nb_bits=4),
    Conv("conv3_1x1", out_mv(4), out_mv(4)),
    BatchNormalization("bn3_1x1"),
    SAT_Quant("satq3_1x1", nb_bits=4),

    # Div8
    Conv("conv4_3x3_dw", out_mv(4), out_mv(4), kernels=[3,3], stride=[2,2], padding=[1,1], grouping=out_mv(4)),
    BatchNormalization("bn4_3x3_dw"),
    SAT_Quant("satq4_3x3_dw", nb_bits=4),
    Conv("conv4_1x1", out_mv(4), out_mv(8)),
    BatchNormalization("bn4_1x1"),
    SAT_Quant("satq4_1x1", nb_bits=4),
    Conv("conv5_3x3_dw", out_mv(8), out_mv(8), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(8)),
    BatchNormalization("bn5_3x3_dw"),
    SAT_Quant("satq5_3x3_dw", nb_bits=4),
    Conv("conv5_1x1", out_mv(8), out_mv(8)),
    BatchNormalization("bn5_1x1"),
    SAT_Quant("satq5_1x1", nb_bits=4),

    # Div16
    Conv("conv6_3x3_dw", out_mv(8), out_mv(8), kernels=[3,3], stride=[2,2], padding=[1,1], grouping=out_mv(8)),
    BatchNormalization("bn6_3x3_dw"),
    SAT_Quant("satq6_3x3_dw", nb_bits=4),
    Conv("conv6_1x1", out_mv(8), out_mv(16)),
    BatchNormalization("bn6_1x1"),
    SAT_Quant("satq6_1x1", nb_bits=4),
    Conv("conv7_1_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(16)),
    BatchNormalization("bn7_1_3x3_dw"),
    SAT_Quant("satq7_1_3x3_dw", nb_bits=4),
    Conv("conv7_1_1x1", out_mv(16), out_mv(16)),
    BatchNormalization("bn7_1_1x1"),
    SAT_Quant("satq7_1_1x1", nb_bits=4),
    Conv("conv7_2_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(16)),
    BatchNormalization("bn7_2_3x3_dw"),
    SAT_Quant("satq7_2_3x3_dw", nb_bits=4),
    Conv("conv7_2_1x1", out_mv(16), out_mv(16)),
    BatchNormalization("bn7_2_1x1"),
    SAT_Quant("satq7_2_1x1", nb_bits=4),
    Conv("conv7_3_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(16)),
    BatchNormalization("bn7_3_3x3_dw"),
    SAT_Quant("satq7_3_3x3_dw", nb_bits=4),
    Conv("conv7_3_1x1", out_mv(16), out_mv(16)),
    BatchNormalization("bn7_3_1x1"),
    SAT_Quant("satq7_3_1x1", nb_bits=4),
    Conv("conv7_4_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(16)),
    BatchNormalization("bn7_4_3x3_dw"),
    SAT_Quant("satq7_4_3x3_dw", nb_bits=4),
    Conv("conv7_4_1x1", out_mv(16), out_mv(16)),
    BatchNormalization("bn7_4_1x1"),
    SAT_Quant("satq7_4_1x1", nb_bits=4),
    Conv("conv7_5_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(16)),
    BatchNormalization("bn7_5_3x3_dw"),
    SAT_Quant("satq7_5_3x3_dw", nb_bits=4),
    Conv("conv7_5_1x1", out_mv(16), out_mv(16)),
    BatchNormalization("bn7_5_1x1"),
    SAT_Quant("satq7_5_1x1", nb_bits=4),

    # Div32
    Conv("conv8_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[2,2], padding=[1,1], grouping=out_mv(16)),
    BatchNormalization("bn8_3x3_dw"),
    SAT_Quant("satq8_3x3_dw", nb_bits=4),
    Conv("conv8_1x1", out_mv(16), out_mv(32)),
    BatchNormalization("bn8_1x1"),
    SAT_Quant("satq8_1x1", nb_bits=4),
    Conv("conv9_3x3_dw", out_mv(32), out_mv(32), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(32)),
    BatchNormalization("bn9_3x3_dw"),
    SAT_Quant("satq9_3x3_dw", nb_bits=4),
    Conv("conv9_1x1", out_mv(32), out_mv(32)),
    BatchNormalization("bn9_1x1"),
    SAT_Quant("satq9_1x1", nb_bits=4),

    # Head
    GlobalAveragePool("pool1"),
    Flatten("flatten1"),
    Gemm("fc", out_mv(32), 1000)
]


CHIP_NeurocorgiModel = [

    # Div2
    Conv("conv1", 3, out_mv(1), kernels=[3,3], stride=[2,2], padding=[1,1]),
    Relu("relu1"),
    ScalingFixedPoint("scale1", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv1_3x3_dw", out_mv(1), out_mv(1), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(1)),
    Relu("relu1_3x3_dw"),
    ScalingFixedPoint("scale1_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv1_1x1", out_mv(1), out_mv(2)),
    Relu("relu1_1x1"),
    ScalingFixedPoint("scale1_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),

    # Div4
    Conv("conv2_3x3_dw", out_mv(2), out_mv(2), kernels=[3,3], stride=[2,2], padding=[1,1], grouping=out_mv(2)),
    Relu("relu2_3x3_dw"),
    ScalingFixedPoint("scale2_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv2_1x1", out_mv(2), out_mv(4)),
    Relu("relu2_1x1"),
    ScalingFixedPoint("scale2_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv3_3x3_dw", out_mv(4), out_mv(4), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(4)),
    Relu("relu3_3x3_dw"),
    ScalingFixedPoint("scale3_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv3_1x1", out_mv(4), out_mv(4)),
    Relu("relu3_1x1"),
    ScalingFixedPoint("scale3_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),

    # Div8
    Conv("conv4_3x3_dw", out_mv(4), out_mv(4), kernels=[3,3], stride=[2,2], padding=[1,1], grouping=out_mv(4)),
    Relu("relu4_3x3_dw"),
    ScalingFixedPoint("scale4_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv4_1x1", out_mv(4), out_mv(8)),
    Relu("relu4_1x1"),
    ScalingFixedPoint("scale4_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv5_3x3_dw", out_mv(8), out_mv(8), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(8)),
    Relu("relu5_3x3_dw"),
    ScalingFixedPoint("scale5_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv5_1x1", out_mv(8), out_mv(8)),
    Relu("relu5_1x1"),
    ScalingFixedPoint("scale5_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),

    # Div16
    Conv("conv6_3x3_dw", out_mv(8), out_mv(8), kernels=[3,3], stride=[2,2], padding=[1,1], grouping=out_mv(8)),
    Relu("relu6_3x3_dw"),
    ScalingFixedPoint("scale6_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv6_1x1", out_mv(8), out_mv(16)),
    Relu("relu6_1x1"),
    ScalingFixedPoint("scale6_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv7_1_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(16)),
    Relu("relu7_1_3x3_dw"),
    ScalingFixedPoint("scale7_1_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv7_1_1x1", out_mv(16), out_mv(16)),
    Relu("relu7_1_1x1"),
    ScalingFixedPoint("scale7_1_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv7_2_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(16)),
    Relu("relu7_2_3x3_dw"),
    ScalingFixedPoint("scale7_2_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv7_2_1x1", out_mv(16), out_mv(16)),
    Relu("relu7_2_1x1"),
    ScalingFixedPoint("scale7_2_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv7_3_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(16)),
    Relu("relu7_3_3x3_dw"),
    ScalingFixedPoint("scale7_3_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv7_3_1x1", out_mv(16), out_mv(16)),
    Relu("relu7_3_1x1"),
    ScalingFixedPoint("scale7_3_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv7_4_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(16)),
    Relu("relu7_4_3x3_dw"),
    ScalingFixedPoint("scale7_4_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv7_4_1x1", out_mv(16), out_mv(16)),
    Relu("relu7_4_1x1"),
    ScalingFixedPoint("scale7_4_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv7_5_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(16)),
    Relu("relu7_5_3x3_dw"),
    ScalingFixedPoint("scale7_5_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv7_5_1x1", out_mv(16), out_mv(16)),
    Relu("relu7_5_1x1"),
    ScalingFixedPoint("scale7_5_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),

    # Div32
    Conv("conv8_3x3_dw", out_mv(16), out_mv(16), kernels=[3,3], stride=[2,2], padding=[1,1], grouping=out_mv(16)),
    Relu("relu8_3x3_dw"),
    ScalingFixedPoint("scale8_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv8_1x1", out_mv(16), out_mv(32)),
    Relu("relu8_1x1"),
    ScalingFixedPoint("scale8_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv9_3x3_dw", out_mv(32), out_mv(32), kernels=[3,3], stride=[1,1], padding=[1,1], grouping=out_mv(32)),
    Relu("relu9_3x3_dw"),
    ScalingFixedPoint("scale9_3x3_dw", mode="Symmetric", fractional_bits=16, quant_bits=4),
    Conv("conv9_1x1", out_mv(32), out_mv(32)),
    Relu("relu9_1x1"),
    ScalingFixedPoint("scale9_1x1", mode="Symmetric", fractional_bits=16, quant_bits=4),

    # Head
    # Deactived because the chip doesn't embed head
    # GlobalAveragePool("pool1"),
    # Flatten("flatten1"),
    # Gemm("fc", out_mv(32), 1000),

    # Not sure if it is required
    # ScalingFloatingPoint("scale_fc")
]


##########################################################################
########################### Fusion functions #############################
##########################################################################

def fuse_qat(model_fq, model_fused):

    # Corrective factor
    corr_factor = 1.0

    # Init alpha1 & range1
    alpha1 = 1.0
    range1 = 255.0

    for i in range(len(model_fq)):
        
        # Fuse BatchNorm with Conv according to SAT operator
        if model_fq[i].op_type == "Conv" \
            and model_fq[i+1].op_type == "BatchNormalization" \
            and model_fq[i+2].op_type == "SAT_Quant":

            # First convolution's weights have been quantized in 8bit
            # Other convolutions parameters have been quantized in 4bit
            # Perhaps change the code for something more modular in the future
            # (if range = 1 -> weight_scale = 1)
            weight_scale = math.floor(255.0 / 2.0) if i == 0 else math.floor(15.0 / 2.0)
            bias_scale = weight_scale

            # Get alpha2 & range2
            alpha2 = model_fq[i+2].alpha[0]
            range2 = model_fq[i+2].range

            # Find corresponding i in fused model
            i_fused = next((j for j in range(len(model_fused)) if model_fused[j].name == model_fq[i].name), None)

            for output in range(model_fq[i].weights.shape[0]):

                bn_scale = model_fq[i+1].scales[output]
                bn_bias = model_fq[i+1].biases[output]
                bn_mean = model_fq[i+1].means[output]
                bn_var = model_fq[i+1].variances[output]
                bn_eps = model_fq[i+1].epsilon

                # Calculate factor
                factor = 0.0
                if bn_var > 1.0e-12:
                    factor = bn_scale / math.sqrt(bn_eps + bn_var)

                sign_factor = 1.0 if factor >= 0.0 else -1.0

                # Quantize each weight of the convolution for the specific output
                model_fused[i_fused].weights[output] = sign_factor * np.round(model_fq[i].weights[output] * weight_scale)

                # Apply fusion formulas
                if factor != 0.0:

                    # Bias
                    bias_fused = bn_bias + (model_fq[i].biases[output] - bn_mean) * factor
                    bias_fused = (range1 / alpha1) * (bias_fused / factor)
                    bias_fused = bias_fused * bias_scale * (1.0 / corr_factor) * sign_factor
                    model_fused[i_fused].biases[output] = round(bias_fused)

                    # Scaling
                    scale_fused = (alpha1 / range1) * (range2 / alpha2) * factor
                    scale_fused = scale_fused * (1 / weight_scale) * corr_factor * sign_factor
                    model_fused[i_fused + 2].scaling[output] = round(scale_fused * (1 << model_fused[i_fused + 2].fractional_bits))

                    # Clipping
                    saturate_fused = ((alpha2 * range1) / (factor * alpha1))
                    saturate_fused = saturate_fused * weight_scale * (1.0 / corr_factor) * sign_factor
                    model_fused[i_fused + 2].clipping[output] = round(saturate_fused)

            # Switch alpha & range for fusion propagation to the next layers
            alpha1 = alpha2
            range1 = range2

        # Quantize Fc parameters (No need so far)
        # if model_fq[i].op_type == "Gemm":

        #     # To complete
        #     # So far, just copy weights and biases
        #     model_fused[i].weights = model_fq[i].weights
        #     model_fused[i].biases = model_fq[i].biases

        #     # Check if there is a SAT operator after FC
        #     # to get alpha2 & range2
        #     if model_fq[i+1].op_type == "SAT_Quant":
        #         alpha2 = model_fq[i+1].alpha[0]
        #         range2 = model_fq[i+1].range

        #     # Fc's parameters have been quantized in 8-bit
        #     # Perhaps change this system to adapt it to densenets
        #     weight_scale = math.floor(255.0 / 2.0)
        #     bias_scale = weight_scale




##########################################################################
######################### Models initialization ##########################
##########################################################################

# Compute I/O dimensions
# And load parameters from weights directory
for i, node in enumerate(FQ_NeurocorgiModel):
    if node.name == "conv1":
        node.input_dims = inputs_dimensions
    else:
        node.input_dims = FQ_NeurocorgiModel[i-1].output_dims

    node.compute_dims()
    node.load_from_n2d2(weight_directory)

for i, node in enumerate(CHIP_NeurocorgiModel):
    if node.name == "conv1":
        node.input_dims = inputs_dimensions
    else:
        node.input_dims = CHIP_NeurocorgiModel[i-1].output_dims

    node.compute_dims()

# Copy weights and biases dims in chip model for convs
# Lazy algorithm in O(n^2)
for i, node1 in enumerate(CHIP_NeurocorgiModel):
    if node1.op_type == "Conv":
        for node2 in FQ_NeurocorgiModel:
            if node2.name == node1.name:
                CHIP_NeurocorgiModel[i].weights = np.zeros_like(node2.weights).astype(np.float32)
                CHIP_NeurocorgiModel[i].biases = np.zeros(node2.weights.shape[0]).astype(np.float32)

    if node1.op_type == "ScalingFixedPoint":
        # Supposed that previous conv has been initialized
        CHIP_NeurocorgiModel[i].scaling = np.zeros(CHIP_NeurocorgiModel[i-2].biases.shape).astype(np.float32)
        CHIP_NeurocorgiModel[i].clipping = np.zeros(CHIP_NeurocorgiModel[i-2].biases.shape).astype(np.float32)

    if node1.op_type == "Gemm":
        for node2 in FQ_NeurocorgiModel:
            if node2.name == node1.name:
                CHIP_NeurocorgiModel[i].weights = np.zeros_like(node2.weights).astype(np.float32)
                CHIP_NeurocorgiModel[i].biases = np.zeros_like(node2.biases).astype(np.float32)

# Apply fusion algorithms
fuse_qat(FQ_NeurocorgiModel, CHIP_NeurocorgiModel)

##########################################################################
##################### ONNX builder for FQ model ##########################
##########################################################################

onnx_nodes = []
onnx_inputs_names = []
onnx_inputs = []
onnx_outputs = []
onnx_initializers = []
onnx_value_info = []

outputs_list = ["satq3_1x1", "satq5_1x1", "satq7_5_1x1","satq9_1x1", "fc"]

for i, node in enumerate(FQ_NeurocorgiModel):
    inputs = []
    if node.name == "conv1":
        inputs.append("data")
        onnx_inputs.append(helper.make_tensor_value_info("data", TensorProto.FLOAT, inputs_dimensions))
    else:
        inputs.append(FQ_NeurocorgiModel[i-1].name + "_out")

    # Set up the initializers
    if isinstance(node, BatchNormalization):

        scale_name = node.name + "_scales"
        onnx_initializers.append(numpy_helper.from_array(node.scales, scale_name))
        inputs.append(scale_name)

        biases_name = node.name + "_biases"
        onnx_initializers.append(numpy_helper.from_array(node.biases, biases_name))
        inputs.append(biases_name)

        means_name = node.name + "_means"
        onnx_initializers.append(numpy_helper.from_array(node.means, means_name))
        inputs.append(means_name)

        variances_name = node.name + "_variances"
        onnx_initializers.append(numpy_helper.from_array(node.variances, variances_name))
        inputs.append(variances_name)

    if isinstance(node, Gemm):

        weight_name = node.name + "_weights"
        onnx_initializers.append(numpy_helper.from_array(node.weights, weight_name))
        inputs.append(weight_name)

        biases_name = node.name + "_biases"
        onnx_initializers.append(numpy_helper.from_array(node.biases, biases_name))
        inputs.append(biases_name)

    if isinstance(node, Conv):

        weight_name = node.name + "_weights"
        onnx_initializers.append(numpy_helper.from_array(node.weights, weight_name))
        inputs.append(weight_name)


    onnx_nodes.append(helper.make_node(
        name=node.name,
        op_type=node.op_type,                
        inputs=inputs,
        outputs=[node.name + "_out"],
    ))

    # Add attributes

    if isinstance(node, SAT_Quant):
        onnx_nodes[-1].attribute.append(helper.make_attribute("alpha", node.alpha[0]))
        onnx_nodes[-1].attribute.append(helper.make_attribute("nb_bits", node.nb_bits))

    if isinstance(node, Conv):
        onnx_nodes[-1].attribute.append(helper.make_attribute("dilations", node.dilation))
        onnx_nodes[-1].attribute.append(helper.make_attribute("group", node.grouping))
        onnx_nodes[-1].attribute.append(helper.make_attribute("kernel_shape", node.kernels))
        onnx_nodes[-1].attribute.append(helper.make_attribute("pads", node.padding))
        onnx_nodes[-1].attribute.append(helper.make_attribute("strides", node.stride))

    if isinstance(node, Gemm):
        onnx_nodes[-1].attribute.append(helper.make_attribute("alpha", node.alpha))
        onnx_nodes[-1].attribute.append(helper.make_attribute("beta", node.beta))
        onnx_nodes[-1].attribute.append(helper.make_attribute("transA", node.transA))
        onnx_nodes[-1].attribute.append(helper.make_attribute("transB", node.transB))

    if isinstance(node, BatchNormalization):
        onnx_nodes[-1].attribute.append(helper.make_attribute("epsilon", node.epsilon))
        onnx_nodes[-1].attribute.append(helper.make_attribute("momentum", node.momentum))

    
    # Add outputs
    if node.name in outputs_list:
        onnx_outputs.append(helper.make_tensor_value_info(node.name + "_out", TensorProto.FLOAT, node.output_dims))
    else:
        # Add value info instead
        onnx_value_info.append(helper.make_tensor_value_info(node.name + "_out", TensorProto.FLOAT, node.output_dims))


# Create the graph (GraphProto)
onnx_graph = onnx.helper.make_graph(
    nodes=onnx_nodes,
    initializer=onnx_initializers,
    name=producer_graph,
    inputs=onnx_inputs,
    outputs=onnx_outputs,
    value_info=onnx_value_info
)
# Create the model (ModelProto)
onnx_model = onnx.helper.make_model(
    onnx_graph, 
    producer_name=producer_name, 
    producer_version=producer_version
)

# Change the name of the outputs
change_dict = {
    "satq3_1x1_out": "div4",
    "satq5_1x1_out": "div8",
    "satq7_5_1x1_out": "div16",
    "satq9_1x1_out": "div32"
}

for i, node in enumerate(onnx_model.graph.node):
    for k, input in enumerate(node.input):
        for key in change_dict.keys():
            if input == key:
                onnx_model.graph.node[i].input[k] = change_dict[key]

    for k, output in enumerate(node.output):
        for key in change_dict.keys():
            if output == key:
                onnx_model.graph.node[i].output[k] = change_dict[key]

for i, output in enumerate(onnx_model.graph.output):
    for key in change_dict.keys():
        if output.name == key:
            onnx_model.graph.output[i].name = change_dict[key]


onnx.save(onnx_model, fq_filename)


##########################################################################
#################### ONNX builder for Chip model #########################
##########################################################################

onnx_nodes = []
onnx_inputs_names = []
onnx_inputs = []
onnx_outputs = []
onnx_initializers = []
onnx_value_info = []

outputs_list = ["scale3_1x1", "scale5_1x1", "scale7_5_1x1","scale9_1x1", "fc"]

for i, node in enumerate(CHIP_NeurocorgiModel):
    inputs = []
    if node.name == "conv1":
        inputs.append("data")
        onnx_inputs.append(helper.make_tensor_value_info("data", TensorProto.FLOAT, inputs_dimensions))
    else:
        inputs.append(CHIP_NeurocorgiModel[i-1].name + "_out")

    # Set up the initializers
    if isinstance(node, ScalingFixedPoint):

        scaling_name = node.name + "_scaling"
        onnx_initializers.append(numpy_helper.from_array(node.scaling, scaling_name))
        inputs.append(scaling_name)

        clipping_name = node.name + "_clipping"
        onnx_initializers.append(numpy_helper.from_array(node.clipping, clipping_name))
        inputs.append(clipping_name)

    if isinstance(node, Gemm):

        weight_name = node.name + "_weights"
        onnx_initializers.append(numpy_helper.from_array(node.weights, weight_name))
        inputs.append(weight_name)

        biases_name = node.name + "_biases"
        onnx_initializers.append(numpy_helper.from_array(node.biases, biases_name))
        inputs.append(biases_name)

    if isinstance(node, Conv):

        weight_name = node.name + "_weights"
        onnx_initializers.append(numpy_helper.from_array(node.weights, weight_name))
        inputs.append(weight_name)

        biase_name = node.name + "_biases"
        onnx_initializers.append(numpy_helper.from_array(node.biases, biase_name))
        inputs.append(biase_name)


    onnx_nodes.append(helper.make_node(
        name=node.name,
        op_type=node.op_type,                
        inputs=inputs,
        outputs=[node.name + "_out"],
    ))

    # Add attributes

    if isinstance(node, Conv):
        onnx_nodes[-1].attribute.append(helper.make_attribute("dilations", node.dilation))
        onnx_nodes[-1].attribute.append(helper.make_attribute("group", node.grouping))
        onnx_nodes[-1].attribute.append(helper.make_attribute("kernel_shape", node.kernels))
        onnx_nodes[-1].attribute.append(helper.make_attribute("pads", node.padding))
        onnx_nodes[-1].attribute.append(helper.make_attribute("strides", node.stride))

    if isinstance(node, Gemm):
        onnx_nodes[-1].attribute.append(helper.make_attribute("alpha", node.alpha))
        onnx_nodes[-1].attribute.append(helper.make_attribute("beta", node.beta))
        onnx_nodes[-1].attribute.append(helper.make_attribute("transA", node.transA))
        onnx_nodes[-1].attribute.append(helper.make_attribute("transB", node.transB))

    if isinstance(node, ScalingFixedPoint):
        onnx_nodes[-1].attribute.append(helper.make_attribute("fractionnal_bits", node.fractional_bits))
        onnx_nodes[-1].attribute.append(helper.make_attribute("mode", node.mode))
        onnx_nodes[-1].attribute.append(helper.make_attribute("quant_bits", node.quant_bits))

    # Add outputs
    if node.name in outputs_list:
        onnx_outputs.append(helper.make_tensor_value_info(node.name + "_out", TensorProto.FLOAT, node.output_dims))
    else:
        # Add value info instead
        onnx_value_info.append(helper.make_tensor_value_info(node.name + "_out", TensorProto.FLOAT, node.output_dims))


# Create the graph (GraphProto)
onnx_graph = onnx.helper.make_graph(
    nodes=onnx_nodes,
    initializer=onnx_initializers,
    name=producer_graph,
    inputs=onnx_inputs,
    outputs=onnx_outputs,
    value_info=onnx_value_info
)
# Create the model (ModelProto)
onnx_model = onnx.helper.make_model(
    onnx_graph, 
    producer_name=producer_name, 
    producer_version=producer_version
)

# Change the name of the outputs
change_dict = {
    "scale3_1x1_out": "div4",
    "scale5_1x1_out": "div8",
    "scale7_5_1x1_out": "div16",
    "scale9_1x1_out": "div32"
}

for i, node in enumerate(onnx_model.graph.node):
    for k, input in enumerate(node.input):
        for key in change_dict.keys():
            if input == key:
                onnx_model.graph.node[i].input[k] = change_dict[key]

    for k, output in enumerate(node.output):
        for key in change_dict.keys():
            if output == key:
                onnx_model.graph.node[i].output[k] = change_dict[key]

for i, output in enumerate(onnx_model.graph.output):
    for key in change_dict.keys():
        if output.name == key:
            onnx_model.graph.output[i].name = change_dict[key]


onnx.save(onnx_model, chip_filename)