import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class ImageList(Dataset):
    """Dataset over an explicit (path, label) list with a single transform."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _list_jpgs(class_dir: str):
    files = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))
    if not files:
        raise SystemExit(f"no .jpg files found in {class_dir!r}")
    return files


def build_splits(class_dirs, train_transform, eval_transform,
                 val_frac=0.1, test_frac=0.1, seed=0):
    """Return (train_ds, val_ds, test_ds, class_names).

    `class_dirs` is a list of source folders, one per class. Labels are assigned
    by sorted basename so the mapping is deterministic regardless of input order
    (matches torchvision ImageFolder convention).
    """
    import random

    rng = random.Random(seed)
    class_dirs = sorted(class_dirs, key=lambda d: os.path.basename(os.path.normpath(d)))

    train, val, test, class_names = [], [], [], []
    for label, d in enumerate(class_dirs):
        class_names.append(os.path.basename(os.path.normpath(d)))
        files = _list_jpgs(d)
        rng.shuffle(files)

        n = len(files)
        n_test = round(n * test_frac)
        n_val = round(n * val_frac)
        test += [(f, label) for f in files[:n_test]]
        val += [(f, label) for f in files[n_test:n_test + n_val]]
        train += [(f, label) for f in files[n_test + n_val:]]

    return (
        ImageList(train, train_transform),
        ImageList(val, eval_transform),
        ImageList(test, eval_transform),
        class_names,
    )
