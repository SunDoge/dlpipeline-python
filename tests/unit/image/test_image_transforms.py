from dlpipeline.image import transforms
from dlpipeline.image.transforms import functional as F
import dlpipeline


def load_dog1():
    img = F.pyvips_loader(dlpipeline.test.image.DOG1)
    return img


def test_load_image():
    img = load_dog1()
    assert img.bands == 3


def test_resize_fn():
    img = load_dog1()
    w, h = 256, 256
    out = F.resize(img, (h, w))
    assert out.width == w, out
    assert out.height == h, out

    out = F.resize(img, h)
    assert out.height == h, out
    assert out.width == round(h * img.width / img.height), out
