import torch
from typing import *
import pyvips
from torch import Tensor
from dlpipeline.image.transforms.functional import (
    numpy_to_vips_image,
    vips_image_to_numpy,
)
import numpy as np


def numpy_image_to_tensor(img: np.ndarray):
    """
    TODO check dtype
    :param img:
    :return:
    """
    tensor = torch.from_numpy(
        img.transpose((2, 0, 1))
    )
    return tensor


def tensor_to_vips_image():
    pass


def vips_image_to_tensor(img: pyvips.Image):
    if img.format == 'uchar':
        tensor = torch.ByteTensor(
            torch.ByteStorage.from_buffer(img.write_to_memory())
        )
        tensor = tensor.view(
            img.height, img.width, img.bands
        )

    else:
        np_img = vips_image_to_numpy(img)

        if np_img.ndim == 2:
            np_img = np_img[:, :, None]

        tensor = torch.from_numpy(np_img)

    tensor = tensor.permute((2, 0, 1)).contiguous()

    if isinstance(tensor, torch.ByteTensor):
        return tensor.float().div(255)
    else:
        return tensor


def normalize(
        tensor: Tensor,
        mean: Sequence[float],
        std: Sequence[float],
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
    if not torch.is_tensor(tensor):
        raise TypeError(
            'tensor should be a transforms tensor. Got {}.'.format(type(tensor)))

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
