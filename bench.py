import cv2
from pathlib import Path
import torch
import numpy as np
from sklearn.utils import shuffle
import onnxruntime as ort
from datasets.suas import rgb2gray
import torch
from string import digits, ascii_uppercase




def show(img, name = "window"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.resizeWindow(name, 500, 500)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(img, size=200):
    return cv2.resize(img, (size, size))


if __name__ == "__main__":
    # path = Path("/home/ubuntu/Ascend/datasets/outputs_cropped")
    #
    # images = list()
    # for image_path in (path / "images").iterdir():
    #     images.append(cv2.imread(image_path.__str__()))
    # 
    # local_images, labels = rgb2gray(images, ['0']*len(images))
    #
    # for local_image, image in zip(local_images, images):
    #     show(image)
    #     show(np.array(local_image))
    # exit()


        



    # images, labels = torch.load(Path("val.mnt"))
    images, labels = torch.load(Path("labeled.mnt"))

    images, labels = shuffle(images, labels) #pyright: ignore


    session = ort.InferenceSession("gcnn-gray.onnx")
    for image, label in zip(images, labels):
        gray_image, local_label = rgb2gray([image], [label])
        gray_image, image = np.array(gray_image).transpose(1,2,0), cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray_image, image = resize(gray_image), resize(image)
        # print(gray_image.reshape(1,1,200,200).astype(np.float32))
        show(image)
        # show(image[40:160,40:160,:])
        outs = session.run(["output"],{"input": gray_image.reshape(1,1,200,200).astype(np.float32) / 255.0})
        topk_values, indicies = torch.topk(torch.tensor(outs[0]), 5)
        topk_values = torch.softmax(topk_values, dim=1)
        name = np.vectorize(lambda x: list(digits+ascii_uppercase)[x])(indicies).flatten()
        confs = topk_values[0].tolist()
        show(resize(gray_image), name=",".join([ f"{n} {c:.2f}"for n, c in zip(name, confs)]))
        # image, local_image = np.array(image), np.array(local_image)
        # show(cv2.GaussianBlur(dilated_image, (9,9), 0))
        # show(image)





        # contours, hierarchy = cv2.findContours(np.array(image), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # for index in range(len(contours)):
        #     cnt=contours[index]
        #     x,y,w,h = cv2.boundingRect(cnt)
        #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        # show(image)
