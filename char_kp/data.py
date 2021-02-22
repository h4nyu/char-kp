import json, torch
from typing import Any
import albumentations as A
from pathlib import Path
from torch import Tensor
from albumentations.pytorch.transforms import ToTensorV2
from object_detection.transforms import inv_normalize
from object_detection import Points, Labels, Image, resize_points
from skimage.io import imread

from torch.utils.data import Dataset
from char_kp import config

keypoint_params = A.KeypointParams(format="xy", label_fields=["labels"])


test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=config.image_width),
        A.PadIfNeeded(
            min_width=config.image_width, min_height=config.image_height, border_mode=0
        ),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ],
    keypoint_params=keypoint_params,
)

train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=max(config.image_width, config.image_height)),
        A.PadIfNeeded(min_width=config.image_width, min_height=config.image_height),
        A.Rotate(limit=(-5, 5), p=1.0, border_mode=0),
        A.OneOf(
            [
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ],
            p=0.2,
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            ],
            p=0.3,
        ),
        A.HueSaturationValue(
            p=0.3,
            hue_shift_limit=15,
            sat_shift_limit=20,
            val_shift_limit=15,
        ),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ],
    keypoint_params=keypoint_params,
)


def read_train_rows(annt_path: str) -> dict[str, Any]:
    with open(annt_path) as f:
        return json.load(f)


def clip_points(points: Points, w: int, h: int) -> Points:
    if len(points) == 0:
        return points
    x, y = points.unbind(-1)
    x = x.clip(0, w)
    y = y.clip(0, h)
    return Points(torch.stack((x, y), dim=-1))


class CharDataset(Dataset):
    def __init__(
        self,
        rows: dict[str, Any],
        transforms: Any,
        image_dir: str = "/store/images",
        num_classes: int = config.num_classes,
    ) -> None:
        self.rows = rows
        self.keys = list(rows.keys())
        self.transforms = transforms
        self.image_dir = Path(image_dir)
        self.num_classes = num_classes

    def __getitem__(self, idx: int) -> tuple[str, Image, Points, Labels]:
        id = self.keys[idx]
        row = self.rows[id]
        path = self.image_dir.joinpath(f"{id}.jpg")
        image = imread(path)
        h, w, _ = image.shape
        points = Points(torch.tensor(row))
        points = resize_points(points, scale_y=h, scale_x=w)
        labels = torch.zeros(len(points))
        transed = self.transforms(image=image, keypoints=points, labels=labels)
        t_image = transed["image"]
        t_labels = Labels(torch.tensor(transed["labels"]))
        t_points = resize_points(
            Points(torch.tensor(transed["keypoints"])),
            scale_x=1 / t_image.shape[2],
            scale_y=1 / t_image.shape[1],
        )
        return (
            id,
            Image(transed["image"]),
            t_points,
            t_labels,
        )

    def __len__(self) -> int:
        return len(self.rows)
