from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from typed_args import TypedArgs, dataclass, add_argument
from tqdm import tqdm

ROOT = '/Users/sundoge/Code/python/dlpipeline/examples/data/images'


@dataclass
class Args(TypedArgs):
    dlp: bool = add_argument('--dlp', action='store_true')
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
    else:
        get_ds_fn = get_torchvision_pipeline

    dl = get_dataloader(get_ds_fn, args)

    for data in tqdm(dl):
        pass
