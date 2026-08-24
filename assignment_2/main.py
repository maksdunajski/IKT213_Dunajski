import cv2
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
image = cv2.imread(os.path.join(HERE, "iris-1.jpg"))
emptyPictureArray = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

def padding(image, border_width):
    # Create a broder around the original image which reflects the
    # edges of the original image
    image_with_border = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite("solutions/padding.png", image_with_border)


def crop(image, x_0, y_0, x_1, y_1):
    copped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("solutions/cropped.png", copped_image)

def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite("solutions/resized.png", resized_image)

def copy(image, emptyPictureArray):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            emptyPictureArray[y, x] = image[y, x]
    cv2.imwrite("solutions/copy.png", emptyPictureArray)

def greyscale(image):
    greyscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("solutions/greyscale.png", greyscaled_image)

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("solutions/hsv.png", hsv_image)

def hue_shifted(image, emtpyPictureArray, hue):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            emptyPictureArray[y, x] = image[y, x] + hue
    cv2.imwrite("solutions/hue_shifted.png", emptyPictureArray)

def smoothing(image):
    ksize = (15,15)
    blurred_image = cv2.blur(image, ksize)
    cv2.imwrite("solutions/blurred.png", blurred_image)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    else:
        print("Invalid rotation angle")
    cv2.imwrite("solutions/rotated.png", cv2.rotate(image, cv2.ROTATE_180))


padding(image, 100)
crop(image, 200, 200, 670, 470)
resize(image, 200, 200)
copy(image, emptyPictureArray)
greyscale(image)
hsv(image)
hue_shifted(image, emptyPictureArray, 50)
smoothing(image)
rotation(image, 90)