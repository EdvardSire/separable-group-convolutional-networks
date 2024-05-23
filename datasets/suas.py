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
from pickle import Pickler, Unpickler

alphabet = string.digits+string.ascii_uppercase
new_alphabet = "012345678ACDEFGHIJKMNPQRTUVXY"
mapping = new_alphabet

def sizeEstimate(lst: list) -> int:
    return sum(sys.getsizeof(x.tobytes()) for x in lst if type(x) == Image.Image)


def rgb2gray(images: list[tImage], labels: list[int], use_tqdm = True):
    local_images = list()
    local_labels = list()
    wrapper = tqdm if use_tqdm else lambda x: x
    for image, label in wrapper(zip(images, labels)):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        dilated_image = cv2.dilate(cv2.Canny(image, 100, 100), cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        local_images.append(Image.fromarray(dilated_image))
        local_labels.append(label)
    return (local_images, local_labels)


def save_pickle(data, save_path: Path, useTorch=True):
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)

    if useTorch:
        torch.save(data, save_path.with_suffix(".mnt"))
    else:
        Pickler(open(save_path.with_suffix(".pkl"), "wb")).dump(data)

def load_pickle(read_path: Path, useTorch=True):
    if useTorch:
        return torch.load(read_path)
    else: 
        return Unpickler(open(read_path, "rb")).load()


class SuasDataset(VisionDataset):
    def __init__(self,
                 label_key = "id_shape",
                 symbol_key = "id_symbol",
                 dataset_root_path: Path = Path("/home/ascend/repos/datasets/custom_new_data"),
                 save_root_path: Path = Path("/home/ascend/repos/datasets/custom_new_data_ocr"),
                 train_mode: bool = True,
                 transform = None,
                 isMultiLabelFeatures = False,
                 pickle_suffix = ".mnt"
                 ):
        super().__init__(transform=transform)
        self.PATH_STEM = (Path("train") if train_mode else Path("val"))
        self.images = list()
        self.labels = list()
        self.dataset_pickle_path = (save_root_path / self.PATH_STEM.with_suffix(pickle_suffix))
        self.label_key = label_key
        self.symbol_key = symbol_key
        self.isMultiLabelFeatures = isMultiLabelFeatures

        if not self.dataset_pickle_path.exists():
            print(f"{self.dataset_pickle_path} not found, generating it!")
            self.prepare(dataset_root_path)
            save_pickle((self.images, self.labels), self.dataset_pickle_path, True if pickle_suffix==".mnt" else False)
        else:
            print(f"{self.dataset_pickle_path} found, loading it!")
            self.images, self.labels = load_pickle(self.dataset_pickle_path, True if pickle_suffix==".mnt" else False)


    def prepare(self, dataset_root_path: Path):
        for i, label_json in enumerate(tqdm((dataset_root_path / self.PATH_STEM).iterdir())):
            if not label_json.suffix == ".json": continue

            with open(label_json.__str__(), 'r') as file:
                json_dump = json.load(file)
                with Image.open((dataset_root_path / self.PATH_STEM / "images" / label_json.stem).with_suffix(".png").__str__()) as image:
                    for index in range(len(json_dump)):
                        if json_dump[index]["label"] == "standard":
                            bbox = json_dump[index]["bbox"]
                            cropped_image = image.convert("RGB").crop((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]))
                            self.images.append(cropped_image)
                            self.labels.append((json_dump[index][self.label_key], json_dump[index][self.symbol_key]))
            if i % 100 == 0:
                print(sizeEstimate(self.images) // 10**6, "MB")


    def saveGray(self, method="manual_kernel"):
        if method == "manual_kernel":
            save_pickle(rgb2gray(self.images, self.labels), self.dataset_pickle_path.with_name(self.PATH_STEM.__str__()+"_gray"))
            return

        elif method == "otsu":
            local_images = list()
            for local_image in self.images:
                local_images.append(
                        cv2.threshold(
                            cv2.GaussianBlur(
                                cv2.cvtColor(np.array(local_image), cv2.COLOR_BGR2GRAY),
                                (5, 5), 0
                            ),
                            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )[1]
                )
            save_pickle((local_images, self.labels), self.dataset_pickle_path.with_name(self.PATH_STEM.__str__()+"_otsu"), useTorch=False)
            return
        

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        if self.label_key == "id_symbol":
            alphabet = string.digits+string.ascii_uppercase
            d = {k: k for k in (string.digits+string.ascii_uppercase)}
            d["O"] = "0"
            d["9"] = "6"
            d["S"] = "5"
            d["W"] = "M"
            d["Z"] = "N"
            d["B"] = "8"
            d["L"] = "7"

            img = self.images[index]
            if self.isMultiLabelFeatures:
                label = int(self.labels[index][2]) 
            else:
                label = int(self.labels[index])

            new_label = new_alphabet.index(d[f"{(alphabet)[label]}"])
            # print(np.vectorize(lambda x: new_alphabet[x])(new_label), np.vectorize(lambda x: alphabet[x])(label))
            if self.transform is not None:
                img = self.transform(img)

            return (img, new_label)
        else:
            assert self.label_key == "id_shape"
            img, label = self.images[index], int(self.labels[index])

            if self.transform is not None:
                img = self.transform(img)

            print(label)

            return (img, label)


if __name__ == "__main__":
    # dataset = SuasDataset(train_mode=True)
    dataset = SuasDataset("id_shape", "id_symbol", train_mode=False, pickle_suffix=".pkl")
    dataset.saveGray("otsu")
    # for i in range(20*20):
    #     dataset.__getitem__(i)
    # dataset.rgb2gray()
