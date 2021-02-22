import base64, requests, os, json
from pathlib import Path
from io import BytesIO
from urllib.parse import urljoin
from PIL import Image
from toolz.curried import groupby, pipe, filter, valfilter, valmap, map


def download_all_images(url: str, out_dir: str) -> None:
    Path(out_dir).mkdir(exist_ok=True)
    imgs_res = requests.post(
        urljoin(url, "/api/v1/image/filter"),
        json=dict(state="Done"),
    )
    imgs_res.raise_for_status()
    imgs = imgs_res.json()
    points_res = requests.post(
        urljoin(url, "/api/v1/point/filter"),
        json={},
    )
    points_res.raise_for_status()
    points = pipe(
        points_res.json(),
        filter(lambda x: x["isGrandTruth"]),
        map(
            lambda x: {
                "point": [x["x"], x["y"]],
                "imageId": x["imageId"],
            }
        ),
        groupby(lambda x: x["imageId"]),
        valmap(lambda dv: pipe(dv, map(lambda x: x["point"]), list)),
    )
    with open(f"{out_dir}/points.json", "w") as f:
        json.dump(points, f)

    Path(f"{out_dir}/images").mkdir(exist_ok=True)
    for img in imgs:
        id = img["id"]
        img_res = requests.post(
            urljoin(url, "/api/v1/image/find"),
            json={"id": id},
        )
        img_res.raise_for_status()
        data = img_res.json()["data"]
        img_points = points.get(id)
        if img_points is None:
            continue
        imgdata = base64.decodebytes(data.encode("ascii"))
        pil_img = Image.open(BytesIO(imgdata)).convert("RGB")
        pil_img.save(f"{out_dir}/images/{id}.jpg", format="JPEG")


if __name__ == "__main__":
    download_all_images(os.environ["STORE_URL"], "/store")
