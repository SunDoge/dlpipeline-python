from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from typed_args import TypedArgs, dataclass, add_argument
from tqdm import tqdm

ROOT = 'examples/data/images'


@dataclass
class Args(TypedArgs):
    dlp: bool = add_argument('--dlp', action='store_true')
    tfds: bool = add_argument('--tfds', action='store_true')
    batch_size: int = add_argument('-b', '--batch-size', default=128)
    num_workers: int = add_argument('-n', '--num-workers', default=2)
    # root: str = add_argument('-r', '--root', default=ROOT)


def get_torchvision_pipeline():
    from torchvision import transforms as T

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3)
    ])

    ds = ImageFolder(
        ROOT,
        transform=transform
    )

    return ds


def get_dlpipeline_pipeline():
    from dlpipeline.image import transforms as T
    from dlpipeline.image.transforms import functional as F

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTorchTensor(),
        T.TorchTensorNormalize([0.5] * 3, [0.5] * 3)
    ])

    ds = ImageFolder(
        ROOT,
        transform=transform,
        loader=F.pyvips_loader
    )

    return ds


def get_tfds_pipeline(args: Args):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    image_folder = get_torchvision_pipeline()
    # samples = []
    samples = image_folder.samples

    ds = tf.data.Dataset.from_generator(lambda: samples, (tf.string, tf.int64))

    def read_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (256, 256))

        return img, label

    ds = ds.repeat(100)

    ds = ds.map(read_image, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(args.batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds


def get_dataloader(get_ds_fn, args: Args):
    ds = get_ds_fn()
    repeat_ds = [ds] * 100
    concat_ds = ConcatDataset(repeat_ds)

    dl = DataLoader(
        concat_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return dl


if __name__ == "__main__":
    args: Args = Args.from_args()
    if args.dlp:
        get_ds_fn = get_dlpipeline_pipeline
    elif args.tfds:
        get_ds_fn = get_tfds_pipeline
    else:
        get_ds_fn = get_torchvision_pipeline

    if not args.tfds:
        dl = get_dataloader(get_ds_fn, args)
    else:
        dl = get_ds_fn(args)

    for data in tqdm(dl):
        pass
