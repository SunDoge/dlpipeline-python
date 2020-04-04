import pyvips
import logging
from pyvips import Image
from typing import *
import numpy as np
from enum import Enum

FORMAT_TO_DTYPE = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

DTYPE_TO_FORMAT = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

logger = logging.getLogger(__name__)


class Kernel:
    LINEAR = 'linear'
    CUBIC = 'cubic'


def pyvips_loader(path: str, access=pyvips.Access.SEQUENTIAL) -> Image:
    return Image.new_from_file(path, access=access)


def pyvips_resize_by_size(img: Image, size: Tuple[int, int], kernel: str = Kernel.LINEAR) -> Image:
    """
    size -> height, width
    """
    # w, h = size
    h, w = size
    scale = w / img.width
    vscale = h / img.height
    return img.resize(scale, vscale=vscale, kernel=kernel)


def resize(img: Image, size: Union[int, Tuple[int, int]], kernel: str = Kernel.LINEAR):
    """
    (h, w)
    """

    if isinstance(size, int):
        w, h = img.width, img.height

        if (w <= h and w == size) or (h <= w and h == size):
            return img

        # resize shorter edge
        if w < h:
            scale = size / w
        else:
            scale = size / h

        return img.resize(scale, kernel=kernel)

    else:
        return pyvips_resize_by_size(img, size, kernel=kernel)


def rescale(img: Image, scale: Union[float, Tuple[float, float]], kernel: str = Kernel.LINEAR) -> Image:
    if isinstance(scale, float):
        return img.resize(scale, kernel=kernel)
    else:
        scale, vscale = scale
        return img.resize(scale, vscale=vscale, kernel=kernel)


def crop(img: Image, top: int, left: int, height: int, width: int) -> Image:
    pass


def image_to_numpy(img: Image) -> np.ndarray:
    """
    https://libvips.github.io/pyvips/intro.html#numpy-and-pil
    """

    np_3d = np.ndarray(
        buffer=img.write_to_memory(),
        dtype=FORMAT_TO_DTYPE[img.format],
        shape=[img.height, img.width, img.bands]
    )
    return np_3d


def numpy_to_image(np_3d: np.ndarray) -> Image:
    """
    https://libvips.github.io/pyvips/intro.html#numpy-and-pil
    """
    height, width, bands = np_3d.shape
    linear = np_3d.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(
        linear.data, width, height, bands,
        DTYPE_TO_FORMAT[str(np_3d.dtype)]
    )
    return vi
