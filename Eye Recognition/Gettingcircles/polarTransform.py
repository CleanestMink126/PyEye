import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt

def calculateRadius(x,y):
    return int(math.sqrt(x**2 + y**2))

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def polarToCart(gray_img, center_x=134, center_y=150, radius = (0,0),output_size = (360,80), filterImg = False):
    '''
    Polor to Cartesian Transform function. Works with both squares
    and rectangles. x,y center and radius can be defined by the user
    if desired.
    '''
    # if gray_img is not None:
    #     pass
    # else:
    #     gray_img = cv2.imread(path,0)
    # else:
    #     img = cv2.imread(path)
    #     # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(gray_img.shape)

    angle = 360
    width = gray_img.shape[0]
    widthTemp = gray_img.shape[1]
    if width != widthTemp:
        width = min(width, widthTemp)

    # Define key variables if not defined already
    if radius[1] == 0:
        radius[1] = calculateRadius(width / 2, width / 2) + 1
    if center_x == 0:
        center_x = width / 2
    if center_y == 0:
        center_y = widthTemp / 2
    new_img = np.zeros((radius[1]-radius[0], angle),dtype=np.float32)

    for r in range(radius[0],radius[1]):
        for theta in range(angle):
            x = center_x + int(r * math.cos(math.radians(theta)))
            y = center_y + int(r * math.sin(math.radians(theta)))
            if x>=0 and x<width and y>=0 and y<width:
                pixelValue = gray_img[x][y]
                new_img[r-radius[0]][theta] = pixelValue

    # new_img = np.transpose(new_img)
    print(new_img)
    print(new_img.shape)
    # cv2.imshow('detected circles',new_img/255)
    # cv2.waitKey(0)

    new_img = cv2.resize(new_img, output_size, interpolation = cv2.INTER_LINEAR)
    # cv2.imshow('detected circles',new_img/255)
    # cv2.waitKey(0)

    # top_trunc = 0
    # bot_trunc = 0
    # switch_flag = True
    # for i in range(len(new_img)):
    #     black_flag = True
    #     for j in range(len(new_img[i])):
    #         if new_img[i][j] != 0:
    #             black_flag = False
    #     if black_flag:
    #         if switch_flag:
    #             top_trunc = i
    #         else:
    #             bot_trunc = i
    #             break
    #     else:
    #         if switch_flag:
    #             switch_flag = False
    # new_img = new_img[top_trunc:bot_trunc]
    if filterImg:
        filters = build_filters()
        filtered_img = process(new_img,filters) # https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
        cvtscale_img = cv2.convertScaleAbs(filtered_img)
        ret, thresh_img= cv2.threshold(cvtscale_img, 127,255, cv2.THRESH_BINARY)
        # plt.figure(1)
        # plt.imshow(gray_img, cmap='gray')
        # plt.figure(2)
        # plt.imshow(new_img, cmap='gray')
        cv2.imshow('detected circles',new_img/255)
        cv2.waitKey(0)
        cv2.imshow('detected circles',filtered_img/255)
        cv2.waitKey(0)
        # cv2.imshow('detected circles',thresh_img/255)
        # cv2.waitKey(0)
        return filtered_img
        # plt.figure(3)
        # plt.imshow(filtered_img, cmap='gray')
        # plt.figure(4)
        # plt.imshow(thresh_img, cmap='gray')
        # plt.show()
    return new_img

if __name__ == "__main__":
    path = 'iris0.png'
    polarToCart(path)
