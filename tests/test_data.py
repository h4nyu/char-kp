from char_kp.data import CharDataset, train_transforms, read_train_rows, test_transforms
from object_detection.utils import DetectionPlot
from object_detection.transforms import inv_normalize


def test_dataset() -> None:
    rows = read_train_rows("/store/points.json")
    dataset = CharDataset(rows=rows, transforms=test_transforms)
    for i in range(10):
        id, img, points, labels = dataset[2]
        plot = DetectionPlot(inv_normalize(img))
        plot.draw_points(points, color="red", size=0.5)
        plot.save(f"/store/test-{i}.jpg")
