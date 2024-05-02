from torchvision.datasets import VisionDataset
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
from PIL.Image import Image as tImage
import torch
import cv2
import numpy as np
import sys
import string


new_alphabet = "012345678ACDEFGHIJKMNPQRTUVXY"
mapping = new_alphabet

def sizeEstimate(lst: list) -> int:
    return sum(sys.getsizeof(x.tobytes()) for x in lst if type(x) == Image.Image)


def rgb2gray(images: list[tImage], labels: list[tImage]):
    local_images = list()
    local_labels = list()
    for image, label in tqdm(zip(images, labels)):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        dilated_image = cv2.dilate(cv2.Canny(image, 100, 100), cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        local_images.append(Image.fromarray(dilated_image))
        local_labels.append(label)
    return (local_images, local_labels)


class SuasDataset(VisionDataset):
    def __init__(self,
                 dataset_root_path: Path = Path("/home/ascend/repos/datasets"),
                 save_root_path: Path = Path("/home/ascend/repos/datasets"),
                 train_mode: bool = True,
                 transform = None
                 ):
        super().__init__(transform=transform)
        self.PATH_STEM = (Path("train_gray") if train_mode else Path("val_gray"))
        self.images = list()
        self.labels = list()
        self.dataset_picke_path = (save_root_path / self.PATH_STEM.with_suffix(".mnt"))

        if not self.dataset_picke_path.exists():
            print(f"{self.dataset_picke_path} not found, generating it!")
            self.prepare(dataset_root_path)
            torch.save((self.images, self.labels), (save_root_path / self.PATH_STEM.with_suffix(".mnt")))
        else:
            print(f"{self.dataset_picke_path} found, loading it!")
            self.images, self.labels = torch.load(self.dataset_picke_path)


    def prepare(self, dataset_root_path: Path):
        for i, label_json in enumerate(tqdm((dataset_root_path / self.PATH_STEM.with_suffix(".mnt")).iterdir())):
            if not label_json.suffix == ".json": continue

            with open(label_json.__str__(), 'r') as file:
                json_dump = json.load(file)
                with Image.open((dataset_root_path / self.PATH_STEM / "images" / label_json.stem).with_suffix(".png").__str__()) as image:
                    for index in range(len(json_dump)):
                        if json_dump[index]["label"] == "standard":
                            bbox = json_dump[index]["bbox"]
                            cropped_image = image.convert("RGB").crop((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]))
                            self.images.append(cropped_image)
                            self.labels.append(json_dump[index]["id_symbol"])

            if i % 100 == 0:
                print(sizeEstimate(self.images) // 10**6, "MB")


    def prepareGray(self):
        torch.save(rgb2gray(self.images, self.labels), self.dataset_picke_path.with_name(self.PATH_STEM.__str__()+"_gray").with_suffix(".mnt"))




        
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        alphabet = string.digits+string.ascii_uppercase
        d = {k: k for k in (string.digits+string.ascii_uppercase)}
        d["O"] = "0"
        d["9"] = "6"
        d["S"] = "5"
        d["W"] = "M"
        d["Z"] = "N"
        d["B"] = "8"
        d["L"] = "7"

        img, label = self.images[index], int(self.labels[index])
        new_label = new_alphabet.index(d[f"{(alphabet)[label]}"])
        # print(np.vectorize(lambda x: new_alphabet[x])(new_label), np.vectorize(lambda x: alphabet[x])(label))
        if self.transform is not None:
            img = self.transform(img)

        return (img, new_label)


if __name__ == "__main__":
    # dataset = SuasDataset(train_mode=True)
    a = Path("/home/ubuntu/Ascend/separable-group-convolutional-networks")
    b = Path("/home/ubuntu/Ascend/separable-group-convolutional-networks")
    dataset = SuasDataset(a, b, train_mode=False)
    # for i in range(20*20):
    #     dataset.__getitem__(i)
    # dataset.rgb2gray()
