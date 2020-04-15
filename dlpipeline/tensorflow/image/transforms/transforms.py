import tensorflow as tf
from typing import *
# from tensorflow.image import ResizeMethod
from . import functional as F
import numbers


class Compose:

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img: tf.Tensor):
        for t in self.transforms:
            img = t(img)
        return img


class Resize:

    def __init__(self, size: Union[int, Tuple[int, int]], method=tf.image.ResizeMethod.BILINEAR):
        self.size = size
        self.method = method

    def __call__(self, img: tf.Tensor):
        return F.resize(img, self.size, method=self.method)


class CenterCrop:

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img: tf.Tensor):
        return F.center_crop(img, self.size)


if __name__ == '__main__':
    img = tf.random.uniform([256, 512, 3])

    trans = Compose([
        Resize(256),
        CenterCrop(224)
    ])

    trans = tf.function(trans)

    out = trans(img)

    print(out.shape)
