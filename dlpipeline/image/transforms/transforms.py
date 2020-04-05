from . import functional as F
from .functional import Kernel
from typing import *
import random
from dlpipeline.common.image.compose import Compose


class Resize:

    def __init__(self, size: Union[int, Tuple[int, int]], kernel: str = Kernel.LINEAR):
        assert isinstance(size, int) or (
                isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.kernel = kernel

    def __call__(self, img):
        return F.resize(img, self.size)


class ToTorchTensor:

    def __call__(self, img):
        return F.image_to_torch_tensor(img)


class CenterCrop:

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            size = (size, size)

        self.size = size

    def __call__(self, img):
        return F.center_crop(img, self.size)


class RandomHorizontalFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return F.hflip(img)
        return img


class TorchTensorNormalize:

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return F.torch_tensor_normalize(
            tensor, self.mean, self.std, inplace=self.inplace
        )
