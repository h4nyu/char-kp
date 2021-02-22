from object_detection.centernet import ToPoints, HMLoss
from object_detection.mkmaps import MkPointMaps
from object_detection.backbones.effnet import (
    EfficientNetBackbone,
)
from object_detection.model_loader import (
    ModelLoader,
    BestWatcher,
)
from object_detection.model_loader import WatchMode
from .model import Net
from object_detection.transforms import normalize_std, normalize_mean

num_classes = 2
image_width = 128 * 2
image_height = 128 * 2
## heatmap
sigma = 1.0
confidence_threshold = 0.5

lr = 5e-4
batch_size = 8
out_idx = 4
channels = 128
metric: tuple[str, WatchMode] = ("score", "max")
backbone_id = 3

cls_depth = 1
out_dir = f"/store/kp-{backbone_id}-{cls_depth}"

backbone = EfficientNetBackbone(
    backbone_id, out_channels=channels, pretrained=True
)
net = Net(
    backbone=backbone,
    num_classes=num_classes,
    channels=channels,
    out_idx=out_idx,
    cls_depth=cls_depth,
)
to_points = ToPoints(threshold=confidence_threshold)
mkmaps = MkPointMaps(
    num_classes=num_classes,
    sigma=sigma,
)
hmloss = HMLoss()

model_loader = ModelLoader(
    out_dir=out_dir,
    key="score",
    best_watcher=BestWatcher(mode="min"),
)
