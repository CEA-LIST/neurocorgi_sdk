# NeuroCorgi SDK

[![License](https://img.shields.io/badge/license-CeCILL--C-blue.svg)](LICENSE)

The NeuroCorgi-SDK is a SDK to use the NeuroCorgi model in your own applications. <br>
This SDK is developed inside the Andante project. For more information about the Andante project, go to https://www.andante-ai.eu/.

The SDK provides some versions of the NeuroCorgi circuit which can simulate the behaviour of the models on chip. For more information about the NeuroCorgi ASIC, check the [presentation](https://ai4di.automotive.oth-aw.de/images/EAI-PDF/2022-09-19_EAI_S2_P2-CEA_IvanMiro-Panades.pdf) of [Ivan Miro Panades](https://www.linkedin.com/in/ivanmiro/) at the International Workshop on Embedded Artificial Intelligence Devices, Systems, and Industrial Apllications (EAI).


## Prerequisites

The NeuroCorgi model is powered by [N2D2](https://github.com/CEA-LIST/N2D2). To install the framework, run in your python environment

```
git clone --recursive https://github.com/CEA-LIST/N2D2.git
cd N2D2
pip install .
```

The SDK also provides a [Pytorch](https://github.com/pytorch/pytorch) wrapper for the NeuroCorgi model. <br>
To install this framework, follow the recommendations at https://pytorch.org/.


## Installation

To install the SDK, run in your python environment
```
git clone https://github.com/CEA-LIST/neurocorgi_sdk
cd neurocorgi_sdk
pip install .
```

You will also need the weights of the model to use it. Two versions exist for both Coco and ImageNet versions of the circuit.


Please send an email to [Ivan Miro Panades](ivan.miro-panades@cea.fr) to get the zips (`imagenet_weights.zip` and `coco_weights.zip`).


## Getting Started

You can import the N2D2 version with `NeuroCorgiNet` and use it like this:

```python
from neurocorgi_sdk.models import NeuroCorgiNet

model = NeuroCorgiNet([1, 3, 224, 224], weights_dir="data/imagenet_weights")
```

You can also import the PyTorch version with `NeuroCorgiNet_torch` like this:

```python
from neurocorgi_sdk.models import NeuroCorgiNet_torch

model = NeuroCorgiNet_torch([32, 3, 224, 224], weights_dir="data/imagenet_weights")
```

You can find some [examples](./examples/) of how to integrate the models in your applications.

For instance:
- [ImageNet](./examples/Imagenet.ipynb): uses the N2D2 NeuroCorgi model to perform an ImageNet inference. 
    You need a local installation of the ImageNet Database to make it work. Once installed somewhere, 
    you need to modify the ILSVRC2012_root variable to locate the correct directory.
- [Cifar100_Transfer](./examples/Cifar100_Transfer.ipynb): use the 
    Pytorch NeuroCorgi model to perform a transfer learning on CIFAR100 classification challenge.

To evaluate NeuroCorgi's performances on other tasks, you should use those scripts as a base inspiration.


## Docker

If you wish to work inside a Docker container, you will need to build the image first. <br>
To do so, run the command
```
docker build --pull --rm -f "Dockerfile" -t andante:neurocorgi_sdk "."
```

After building the image, start a container
```
docker run --name myContainer --gpus=all -it andante:neurocorgi_sdk
```


## The Team

The NeuroCorgi-SDK is a project that brought together several skillful engineers and researchers who contributed to it.

This SDK is currently maintained by [Lilian Billod](https://fr.linkedin.com/in/lilian-billod-3737b6177) and [Vincent Templier](http://www.linkedin.com/in/vincent-templier).
A huge thank you to the people who contributed to the creation of this SDK: [Ivan Miro Panades](https://www.linkedin.com/in/ivanmiro/), [Vincent Lorrain](https://fr.linkedin.com/in/vincent-lorrain-71510583), [Inna Kucher](https://fr.linkedin.com/in/inna-kucher-phd-14528169), [David Briand](https://fr.linkedin.com/in/david-briand-a0b1524a), [Johannes Thiele](https://ch.linkedin.com/in/johannes-thiele-51b795130), [Cyril Moineau](https://fr.linkedin.com/in/cmoineau), [Nermine Ali](https://fr.linkedin.com/in/nermineali) and [Olivier Bichler](https://fr.linkedin.com/in/olivierbichler).


## License

This SDK has a CeCill-style license, as found in the [LICENSE](LICENSE) file.