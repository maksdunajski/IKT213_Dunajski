import numpy as np
import cv2

img = cv2.imread('iris-1.jpg',1)


def print_image_information(image):
    height, width, channels = image.shape
    print("Image height:", height)
    print("Image width:", width)
    print("Number of channels:", channels)
    print("Image size: ", image.size)
    print("Image data type:", image.dtype)

def web_camera_information():
    cam = cv2.VideoCapture(0)
    camFPS = cam.get(cv2.CAP_PROP_FPS)
    camHeight = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    camWidth = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    return camFPS, camHeight, camWidth

def save_info_to_file(camInfo):
    with open("solutions/camera_outputs.txt", "w") as f:
        f.write(f"Camera FPS: {camInfo[0]}\n")
        f.write(f"Camera Height: {camInfo[1]}\n")
        f.write(f"Camera Width: {camInfo[2]}\n")


print_image_information(img)
save_info_to_file(web_camera_information())
