import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt

def calculateRadius(x,y):
    return int(math.sqrt(x**2 + y**2))

def polarToCart(path, center_x=0, center_y=0, radius = 0,gray = 1):
    '''
    Polor to Cartesian Transform function. Works with both squares
    and rectangles. x,y center and radius can be defined by the user
    if desired.
    '''
    if gray:
        gray_img = cv2.imread(path,0)
    else:
        img = cv2.imread(path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray_img.shape)

    angle = 360
    width = gray_img.shape[0]
    widthTemp = gray_img.shape[1]
    if width != widthTemp:
        width = min(width, widthTemp)

    # Define key variables if not defined already
    if radius == 0:
        radius = calculateRadius(width / 2, width / 2) + 1
    if center_x == 0:
        center_x = width / 2
    if center_y == 0:
        center_y = widthTemp / 2
    new_img = np.zeros((radius, angle),dtype=int)

    for r in range(radius):
        for theta in range(angle):
            x = center_x + int(r * math.cos(math.radians(theta)))
            y = center_y + int(r * math.sin(math.radians(theta)))
            if x>=0 and x<width and y>=0 and y<width:
                pixelValue = gray_img[x][y]
                new_img[r][theta] = pixelValue

    plt.imshow(new_img, cmap='gray')
    plt.show()
    return new_img

if __name__ == "__main__":
    path = 'iris0.png'
    polarToCart(path)
