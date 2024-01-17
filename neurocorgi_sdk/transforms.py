import torch
from torchvision.transforms import functional as F

class ToNeuroCorgiChip:
    """Convert a Tensor or numpy.ndarray into the format required for proper NeuroCorgi chip operation.

    This transform performs:
    * A check if the input data is a tensor (if not, call ToTensor())
    * A check if the input data is clipped between 0 and 1 (if so, multiplies each element by 255)
    * A check is the input data is between 0 and 255

    """

    def __call__(self, img):
        """
        Args:
            img (Tensor or numpy.ndarray): Image to be converted to NeuroCorgi chip data format.

        Returns:
            NC Image: Image converted to NeuroCorgi chip data format.

        """

        # If img is not a Tensor
        # call ToTensor transformation
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)

        if torch.all((img >= 0) & (img <= 1)):
            img = img * 255
        else:
            if not torch.all((img >= 0) & (img <= 255)):
                # Normalize between 0 and 255
                x_max = torch.max(img)
                x_min = torch.min(img)
                img = ((img - x_min) / (x_max - x_min)) * 255
            
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
