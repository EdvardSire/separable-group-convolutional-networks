import cv2
from pathlib import Path


def show(img, name = "window"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for image_path in Path(".").iterdir():
        if image_path.suffix != ".png": continue
        image = cv2.imread(image_path.__str__())
        show(image)
        local_image = cv2.Canny(image, 100, 100)
        
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        local_image = cv2.dilate(local_image, element)
        show(local_image)
        contours, hierarchy = cv2.findContours(local_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        innermost_contours = []

        for index in range(len(contours)):
            cnt=contours[index]
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        show(image)
