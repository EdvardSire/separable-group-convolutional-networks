from torchvision.datasets import VisionDataset
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
import torch
import sys

def sizeEstimate(lst):
    return sum(sys.getsizeof(x.tobytes()) for x in lst if type(x) == Image.Image)

class SuasDataset(VisionDataset):
    def __init__(self,
                 dataset_root_path: Path = Path("/home/ascend/repos/datasets"),
                 save_root_path: Path = Path("/home/ascend/repos/datasets"),
                 train_mode: bool = True,
                 transform = None
                 ):
        super().__init__(transform=transform)
        PATH_STEM = (Path("train_gray") if train_mode else Path("val_gray"))
        self.images = list()
        self.labels = list()
        self.dataset_picke_path = (save_root_path / PATH_STEM.with_suffix(".mnt"))

        if not self.dataset_picke_path.exists():
            print(f"{self.dataset_picke_path} not found, generating it!")
            self.prepare(dataset_root_path, PATH_STEM=PATH_STEM)
            torch.save((self.images, self.labels), (save_root_path / PATH_STEM.with_suffix(".mnt")))
        else:
            print(f"{self.dataset_picke_path} found, loading it!")
            self.images, self.labels = torch.load(self.dataset_picke_path)


    def prepare(self, dataset_root_path: Path, PATH_STEM: Path):
        for i, label_json in enumerate(tqdm((dataset_root_path / PATH_STEM).iterdir())):
            if not label_json.suffix == ".json": continue

            with open(label_json.__str__(), 'r') as file:
                json_dump = json.load(file)
                with Image.open((dataset_root_path / PATH_STEM / "images" / label_json.stem).with_suffix(".png").__str__()) as image:
                    for index in range(len(json_dump)):
                        if json_dump[index]["label"] == "standard":
                            bbox = json_dump[index]["bbox"]
                            cropped_image = image.convert("RGB").crop((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]))
                            self.images.append(cropped_image)
                            self.labels.append(json_dump[index]["id_symbol"])

            if i % 100 == 0:
                print(sizeEstimate(self.images) // 10**6, "MB")
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index], int(self.labels[index])


        if self.transform is not None:
            img = self.transform(img)

        return (img, label)


if __name__ == "__main__":
    # dataset = SuasDataset(train_mode=True)
    dataset = SuasDataset(train_mode=True)
