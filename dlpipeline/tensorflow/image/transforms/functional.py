import numbers
from typing import *

import tensorflow as tf


def normalize(
        tensor: tf.Tensor,
        mean: Union[int, Tuple[int, int, int]],
        std: Union[int, Tuple[int, int, int]],
) -> tf.Tensor:
    dtype = tensor.dtype
    mean = tf.constant(mean, dtype=dtype)
    std = tf.constant(std, dtype=dtype)

    if tf.reduce_any(std == 0):
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    return (tensor - mean) / std


def resize(
        img: tf.Tensor,
        size: Union[int, Tuple[int, int]],
        method=tf.image.ResizeMethod.BILINEAR
) -> tf.Tensor:
    """
    Resize
    :param img:
    :param size: int for resize shorter edge
    :param method:
    :return:
    """
    if isinstance(size, int):
        h, w, _c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return tf.image.resize(img, (oh, ow), method=method)
        else:
            oh = size
            ow = int(size * w / h)
            return tf.image.resize(img, (oh, ow), method=method)
    else:
        return tf.image.resize(img, size, method=method)


def crop(
        img: tf.Tensor,
        top: int,
        left: int,
        height: int,
        width: int,
):
    return tf.image.crop_to_bounding_box(img, top, left, height, width)


def center_crop(img: tf.Tensor, output_size: Union[int, Tuple[int, int]]) -> tf.Tensor:
    if isinstance(output_size, numbers.Number):
        output_size = (output_size, output_size)

    image_height, image_width, _channels = img.shape
    crop_height, crop_width = output_size
    crop_top = tf.cast(tf.round((image_height - crop_height) / 2.), tf.int32)
    crop_left = tf.cast(tf.round((image_width - crop_width) / 2.), tf.int32)
    return crop(img, crop_top, crop_left, crop_height, crop_width)
