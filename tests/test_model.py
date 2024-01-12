import pytest
import torch
from neurocorgi_sdk import NeuroCorgiNet

def test_model_forward():
    """Test the forward pass of the NeuroCorgiNet model."""
    model = NeuroCorgiNet()
    model(torch.rand(1, 3, 224, 224))
