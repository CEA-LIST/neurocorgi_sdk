# NeuroCorgi SDK

[![NeuroCorgi SDK CI](https://github.com/CEA-LIST/neurocorgi_sdk/actions/workflows/ci.yaml/badge.svg)](https://github.com/CEA-LIST/neurocorgi_sdk/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/license-CeCILL--C-blue.svg)](LICENSE)

The NeuroCorgi-SDK is a SDK to use the NeuroCorgi model in your object detection, instance segmentation and classification applications as a feature extractor. <br>
This SDK is developed inside the Andante project. For more information about the Andante project, go to https://www.andante-ai.eu/.

The SDK provides some versions of the NeuroCorgi circuit which can simulate the behaviour of the models on chip. Two versions have been designed from a MobileNetV1 trained and quantized in 4-bit by using the [SAT](https://arxiv.org/abs/1912.10207) method in [N2D2](https://github.com/CEA-LIST/N2D2): one with the [ImageNet](https://www.image-net.org/index.php) dataset and the second with the [Coco](https://cocodataset.org/#home) dataset.

<div align="center">
    <a href="https://ai4di.automotive.oth-aw.de/images/EAI-PDF/2022-09-19_EAI_S2_P2-CEA_IvanMiro-Panades.pdf#page=17">
    <img src="https://github.com/CEA-LIST/neurocorgi_sdk/raw/master/docs/_static/NeuroCorgi_design.png" width="80%" alt="NeuroCorgi ASIC">
    </a>
    <figcaption>NeuroCorgi ASIC</figcaption>
</div>

NeuroCorgi ASIC is able to extract features from HD images (1280x720) at 30 FPS with less than 100 mW.

<div align="center">
    <a href="https://ai4di.automotive.oth-aw.de/images/EAI-PDF/2022-09-19_EAI_S2_P2-CEA_IvanMiro-Panades.pdf#page=10">
    <img src="https://github.com/CEA-LIST/neurocorgi_sdk/raw/master/docs/_static/NeuroCorgi_performance.png" width="80%" alt="NeuroCorgi performance">
    </a>
    <figcaption>NeuroCorgi performance target</figcaption>
</div>



For more information about the NeuroCorgi ASIC, check the [presentation](https://ai4di.automotive.oth-aw.de/images/EAI-PDF/2022-09-19_EAI_S2_P2-CEA_IvanMiro-Panades.pdf) of [Ivan Miro-Panades](https://www.linkedin.com/in/ivanmiro/) at the International Workshop on Embedded Artificial Intelligence Devices, Systems, and Industrial Applications (EAI).


## Installation

Before installing the sdk package, be sure to have the weight files of the NeuroCorgi models to use them with the sdk. Two versions exist for both Coco and ImageNet versions of the circuit. <br>
Please choose what you want:

| | For PyTorch integration | For N2D2 integration |
|:-:|:-:|:-:|
| ImageNet chip | `neurocorginet_imagenet.safetensors` | `imagenet_weights.zip` |
| Coco chip | `neurocorginet_coco.safetensors` | `coco_weights.zip` |

Please send an email to [Ivan Miro-Panades](ivan.miro-panades@cea.fr) or [Vincent Templier](vincent.templier@cea.fr) to get the files.

### Via PyPI

Pip install the sdk package including all requirements in a [**Python>=3.7**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).
```
pip install neurocorgi_sdk
```


### From Source

To install the SDK, run in your [**Python>=3.7**](https://www.python.org/) environment
```
git clone https://github.com/CEA-LIST/neurocorgi_sdk
cd neurocorgi_sdk
pip install .
```

### With Docker

If you wish to work inside a Docker container, you will need to build the image first. <br>
To do so, run the command
```
docker build --pull --rm -f "docker/Dockerfile" -t neurocorgi:neurocorgi_sdk "."
```

After building the image, start a container
```
docker run --name myContainer --gpus=all -it neurocorgi:neurocorgi_sdk
```


## Getting Started

Use `NeuroCorgiNet` directly in a Python environment:
```python
from neurocorgi_sdk import NeuroCorgiNet
from neurocorgi_sdk.transforms import ToNeuroCorgiChip

# Load a model
model = NeuroCorgiNet("neurocorginet_imagenet.safetensors")

# Load and transform an image (requires PIL and requests)
image = PIL.Image.open(requests.get("https://github.com/CEA-LIST/neurocorgi_sdk/blob/master/neurocorgi_sdk/assets/corgi.jpg", stream=True).raw)
img = ToNeuroCorgiChip()(image)

# Use the model
div4, div8, div16, div32 = model(img)
```

## About NeuroCorgi model

The NeuroCorgi circuit embeds a version of MobileNetV1 which has been trained and quantized in 4-bit. It requires to provide unsigned 8-bit inputs. To respect this condition, **inputs provided to `NeuroCorgiNet` must be between 0 and 255**. <br>
You can use the `ToNeuroCorgiChip` transformation to transform your images in the correct format. (No need to use it with the fakequant version `NeuroCorgiNet_fakequant` as the inputs have to be between 0 and 1).

Moreover, since the model is fixed on chip it is not possible to modify its parameters.
So it will be impossible to train the models but you can train additional models to plug with `NeuroCorgiNet` for your own applications.


## Examples

You can find some [examples](./examples/) of how to integrate the models in your applications.

For instance:
- [ImageNet](./examples/classification/Imagenet.ipynb): use `NeuroCorgiNet` to perform an ImageNet inference. 
    You need a local installation of the ImageNet Database to make it work. Once installed somewhere, 
    you need to modify the ILSVRC2012_root variable to locate the correct directory.
- [Cifar100](./examples/classification/Cifar100.ipynb): use `NeuroCorgiNet` to perform a transfer learning on CIFAR100 classification challenge.

To evaluate NeuroCorgi's performances on other tasks, you should use those scripts as a base inspiration.


## The Team

The NeuroCorgi-SDK is a project that brought together several skillful engineers and researchers who contributed to it.

This SDK is currently maintained by [Lilian Billod](https://fr.linkedin.com/in/lilian-billod-3737b6177) and [Vincent Templier](http://www.linkedin.com/in/vincent-templier).
A huge thank you to the people who contributed to the creation of this SDK: [Ivan Miro-Panades](https://www.linkedin.com/in/ivanmiro/), [Vincent Lorrain](https://fr.linkedin.com/in/vincent-lorrain-71510583), [Inna Kucher](https://fr.linkedin.com/in/inna-kucher-phd-14528169), [David Briand](https://fr.linkedin.com/in/david-briand-a0b1524a), [Johannes Thiele](https://ch.linkedin.com/in/johannes-thiele-51b795130), [Cyril Moineau](https://fr.linkedin.com/in/cmoineau), [Nermine Ali](https://fr.linkedin.com/in/nermineali) and [Olivier Bichler](https://fr.linkedin.com/in/olivierbichler).


## License

This SDK has a CeCill-style license, as found in the [LICENSE](LICENSE) file.