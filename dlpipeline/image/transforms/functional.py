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


def pyvips_loader(path: str, access=pyvips.Access.RANDOM) -> Image:
    """
    Strange performance gain when using Access.RANDOM
    """
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
    return img.crop(left, top, width, height)


def center_crop(img: Image, output_size: Union[int, Tuple[int, int]]):

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    width, height = output_size
    return img.smartcrop(width, height, interesting='centre')


def resized_crop(
    img: Image,
    top: int,
    left: int,
    height: int,
    width: int,
    size: Union[int, Tuple[int, int]],
    kernel: str = Kernel.LINEAR
) -> Image:
    img = crop(img, top, left, height, width)
    img = resize(img, (height, width), kernel=kernel)
    return img


def hflip(img: Image):
    return img.fliphor()


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


def image_to_torch_tensor(img: Image):
    import torch

    if img.format == 'uchar':
        tensor = torch.ByteTensor(
            torch.ByteStorage.from_buffer(img.write_to_memory()))
        tensor = tensor.view(
            img.height, img.width, img.bands
        )

    else:
        np_img = image_to_numpy(img)

        if np_img.ndim == 2:
            np_img = np_img[:, :, None]

        tensor = torch.from_numpy(np_img)

    tensor = tensor.permute((2, 0, 1)).contiguous()

    if isinstance(tensor, torch.ByteTensor):
        return tensor.float().div(255)
    else:
        return tensor


def torch_tensor_normalize(
    tensor,
    mean: Union[float, Tuple[float, float, float]],
    std: Union[float, Tuple[float, float, float]],
    inplace=False
):
    """Normalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.
    Returns:
        Tensor: Normalized Tensor image.
    """
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError(
            'tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndimension() != 3:
        raise ValueError('Expected tensor to be a tensor image of size (C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor
