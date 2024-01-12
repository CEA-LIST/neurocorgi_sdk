# NeuroCorgi SDK 🐕, CeCILL-C license

import cv2
import torch
import pytest
from PIL import Image
from torchvision.transforms import ToTensor

from neurocorgi_sdk.utils import ASSETS
from neurocorgi_sdk.transforms import ToNeuroCorgiChip
from neurocorgi_sdk import NeuroCorgiNet, NeuroCorgiNet_fakequant

IMG_TEST = ASSETS / "corgi.jpg"


def test_model_forward():
    """Test the forward pass of the NeuroCorgiNet model."""
    model = NeuroCorgiNet()
    model(torch.rand(1, 3, 224, 224))


def test_model_fq_forward():
    """Test the forward pass of the NeuroCorgiNet fakequant model."""
    model = NeuroCorgiNet_fakequant()
    model(torch.rand(1, 3, 224, 224))


def test_transform():
    # PIL to tensor
    image = Image.open(IMG_TEST)
    tensor = ToNeuroCorgiChip()(image)
    assert isinstance(tensor, torch.Tensor)
    assert torch.all((tensor >= 0) & (tensor <= 255))

    # Tensor [0;1] to tensor
    image = cv2.imread(str(IMG_TEST))
    tensor = cv2.resize(image, (32, 32))
    tensor = ToTensor()(tensor)
    tensor = ToNeuroCorgiChip()(tensor)
    assert isinstance(tensor, torch.Tensor)
    assert torch.all((tensor >= 0) & (tensor <= 255))

    # Tensor [0;255] to tensor
    image = cv2.imread(str(IMG_TEST))
    tensor = cv2.resize(image, (32, 32))
    tensor = ToTensor()(tensor)
    tensor = tensor * 255
    tensor = ToNeuroCorgiChip()(tensor)
    assert isinstance(tensor, torch.Tensor)
    assert torch.all((tensor >= 0) & (tensor <= 255))

    # Raise error
    with pytest.raises(ValueError):
        tensor = torch.randint(256, 512, (3, 32, 32))
        tensor = ToNeuroCorgiChip()(tensor)

    # Test __repr__
    assert repr(ToNeuroCorgiChip()) == 'ToNeuroCorgiChip()'


