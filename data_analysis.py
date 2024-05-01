from datasets.suas import SuasDataset
import string
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    for t in ["train", "val"]:
        classes = list(string.digits+string.ascii_uppercase)
        classes_freq = list(np.zeros(len(classes)))
        dataset = SuasDataset(train_mode=(t == "train"))

        for class_entry in tqdm(dataset.labels):
            classes_freq[class_entry] += 1

        plt.bar(classes, classes_freq)
        plt.savefig(f"{t}.png")
