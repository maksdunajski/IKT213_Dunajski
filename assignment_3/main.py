import cv2
import numpy as np
from pygame.transform import threshold

img = cv2.imread("images/lambo.png")
template = cv2.imread("images/shapes_template.jpg")
shapes = cv2.imread("images/shapes-1.png")

def sobel_edge_detection(image):
    blurredImage = cv2.GaussianBlur(image, (3,3), 0)
    sobelx = cv2.Sobel(src=blurredImage, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
    sobely = cv2.Sobel(src=blurredImage, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=1)
    sobelxy = cv2.magnitude(sobelx, sobely)
    cv2.imwrite('solutions/task1.png', cv2.convertScaleAbs(sobelxy))

def canny_edge_detections(image, threshold_1, threshold_2):
    blurredImage = cv2.GaussianBlur(image, ksize=(3,3), sigmaX=0)
    edges = cv2.Canny(blurredImage, threshold_1, threshold_2)
    cv2.imwrite('solutions/task2.png', edges)

def template_match(image, template):
    greyscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    greyscaleTemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = greyscaleTemplate.shape[::-1]
    match = cv2.matchTemplate(greyscaleImage, greyscaleTemplate, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    location = np.where(match >= threshold)
    for pt in zip(*location[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imwrite('solutions/task3.png', image)

def resize(image, scale_factor: int, up_or_down: str):
    rows, cols, _channels = map(int, image.shape)
    if up_or_down == "down":
        scaledImage = cv2.pyrDown(image, dstsize=(scale_factor // cols, scale_factor // rows))
        cv2.imwrite("solutions/task4downScaled.png", scaledImage)
    elif up_or_down == "up":
        scaledImage = cv2.pyrUp(image, dstsize=(scale_factor * cols, scale_factor * rows))
        cv2.imwrite("solutions/task4upScaled.png", scaledImage)
    else:
        print("Invalid type")
        return

sobel_edge_detection(img)
canny_edge_detections(img, 50, 50)
template_match(shapes, template)
resize(img, 2, "up")
resize(img, 2, "down")