from dlpipeline.image.transforms import functional as F
import dlpipeline
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np


def load_dog1():
    vimg = F.pyvips_loader(dlpipeline.test.image.DOG1)
    pimg = Image.open(dlpipeline.test.image.DOG1)
    return vimg, pimg


def allclose(vimg, pimg) -> bool:
    np_vimg = F.image_to_numpy(vimg)
    np_pimg = np.array(pimg)
    return np.allclose(np_vimg, np_pimg)


def test_load_image():
    vimg, pimg = load_dog1()
    assert vimg.bands == 3

    assert allclose(vimg, pimg)


def test_resize_fn():
    vimg, pimg = load_dog1()
    w, h = 256, 224
    vout = F.resize(vimg, (h, w))
    assert vout.width == w and vout.height == h

    vout = F.resize(vimg, h)
    pout = TF.resize(pimg, h)
    assert vout.height == pout.height
    assert abs(vout.width - pout.width) <= 1
