import cv2
import numpy as np
from pathlib import Path 


THRESHHOLD = .5

def show(img, name = "window"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def to640(img):
    return cv2.resize(img, (640,640))

def getLocalONNX(path: Path):
    return [elem for elem in Path(path).iterdir() if elem.suffix == ".onnx"]

def xywh2xyxy(x, y, w, h):
    x1 = int (x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)
    return (x1, y1), (x2, y2)


def postprocessV5(outs, img, name="yolov5"):
    outs = np.array(outs[0][0]) # here we get (25200,7)
    # Looking at one row we have (x, y, w, h, objectness  | prob class n)
    outs = outs[outs[:, 4] > THRESHHOLD] 

    for x, y, w, h, obj, prob_e, prob_s in outs:
        draw(img, x, y, w, h)
    show(img, name=name)


def postprocessV8(outs, img, name="yolov8"):
    outs = np.array(outs[0][0]).transpose(1,0) # (8400, 6)

    # x, y, w, h, prob_s, prob_e        
    outs = outs[outs[:, 4] > THRESHHOLD]
    for x, y, w, h, prob_e, prob_s in outs:
        draw(img, x, y, w, h)
    show(img, name=name)

def postprocessShapeDetect(outs, img, name="ShapeDetect"):
      # 0: emergent
      # 1: circle
      # 2: cross
      # 3: pentagon
      # 4: quarter_circle
      # 5: rectangle
      # 6: semicircle
      # 7: star
      # 8: triangle

    outs = np.array(outs[0][0]).transpose(1,0)
    outs = outs[outs[:, 9] > 0.2]
    print(outs.shape)
    for x, y, w, h, e, circle, cross, pentagon, quarter, rect, semi, star, triangle  in outs:
        draw(img, x, y, w, h)
    show(img, name=name)

def draw(img, x, y, w, h):
    p1, p2 = xywh2xyxy(x,y,w,h)
    cv2.rectangle(img, p1, p2, color=(0,255,0), thickness=2)
