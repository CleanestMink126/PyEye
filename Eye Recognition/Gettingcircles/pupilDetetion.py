import cv2
import numpy as np
import os
from operator import itemgetter
import heapq
import matplotlib.pyplot as plt

# def getBorderPixels(image, center, radius, thickness):
#     '''The idea of this function is to return a set of tuples representing x,y
#     coordinates of pixels along the circuference of a given circle'''
#     pass
# def getDataFromMask(mask):
#     '''This might find the pupil given the mask'''

def getdistanceNumpy(center, size):
    x = np.arange(size[1])
    y = np.transpose(np.arange(size[0]))
    z = (x-center[0])**2 + (y[:, np.newaxis]-center[1])**2
    values = [(value, index)for index, value in np.ndenumerate(z)]
    heapq.heapify(values)
    return values

def getNextBand(width, radius, heap):
    numBoxes = round(np.pi * ((2* radius * width) + width**2))
    print(numBoxes)
    return [heapq.heappop(heap) for i in range(int(numBoxes))]

# values = getdistanceNumpy((100,100),(240,480))
width = 3
# for i in range(0, width*20,width):
#     print getNextBand(width, i, values)


def displayImg(filename):
    history = []

    img = cv2.imread('../EyePictures/' + filename,0)
    img = cv2.medianBlur(img,5)
    kernel = np.ones((3,3),np.uint8)



    gray_filtered = cv2.inRange(img, 0, 60)
    mask = cv2.erode(gray_filtered,kernel,iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,2,100,
                                param1=80,param2=60)


    for i in circles[0,:]:
        # draw the outer circle
        # cv2.circle(img,(i[0],i[1]),i[2],255,1, cv2.LINE_AA)
        # # draw the center of the circle
        # cv2.circle(img,(i[0],i[1]),2,255,3)
        center = (i[0],i[1])
        break

    values = getdistanceNumpy(center,img.shape[:2])
    for i in range(0, width*40,width):
        pixels = getNextBand(width, i, values)
        history.append(np.mean([img[pix[1]] for pix in pixels]))
        # for val,place in pixels:
        #     img[place] = 255
        #
        # cv2.imshow('detected circles',img)
        # cv2.waitKey(0)
    plt.plot(history)
    history = []
    plt.show()
    cv2.imshow('detected circles',img)
    cv2.waitKey(0)

directory  = '../EyePictures/'

numEyes = 0
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        displayImg(filename)
    numEyes +=1
    if numEyes > 5: break
        # print(os.path.join(directory, filename))
#
#
# cv2.destroyAllWindows()
