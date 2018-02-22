import cv2
import numpy as np
import os
from operator import itemgetter
import Queue
from pupilDeps import *


# values = getdistanceNumpy((100,100),(240,480))
width = 3
# for i in range(0, width*20,width):
#     print getNextBand(width, i, values)

# def changeValue(img, ):
def examineBaseline(filename):
    '''This will use getbaseline to approximate the standard deviation and mean
    of the pixel values of the eye. Then it will change each point in a certain
    radius to a value approximating how far away from the iris mean'''
    highestBand,radius, pupilRad, values, img, center = getBaseline(filename)
    mean = np.mean(highestBand)
    std = np.std(highestBand)
    history = []
    for i in range(0, width*50,width):
        pixels = getNextBand(width, i, values)
        band =[ ]
        for pix in pixels:
            val = img[pix[1]]
            val = abs(val - mean)/std
            if val > 10:
                val = 10
            if val < 1.5:
                img[pix[1]] = 200
            else:
                img[pix[1]] = 0
            img[pix[1]] = round((10 - val)* 25.5)

    cv2.imshow('detected circles',img)
    cv2.waitKey(0)


def showLateral(filename):
    left,total,right, edge1,edge2 = expandLateral(filename)
    for i,v in enumerate(total):
        img[(int(center[1]-1), int(edge1+ i))] = irisRad2[i]
        img[(int(center[1]-1), int(edge2- i))] = irisRad1[i]
        img[(int(center[1]), int(edge1+ i))] = irisRad1[i]
        img[(int(center[1]), int(edge2- i))] = irisRad2[i]
        img[(int(center[1]+1), int(edge1+ i))] = v
        img[(int(center[1]+1), int(edge2- i))] = v


    cv2.imshow('detected circles',img)
    cv2.waitKey(0)

def getPupil(filename):
    '''the aim of this function is to find the pixels where the border of the
    pupil buts up against the iris'''
    pass


def displayImg(filename):
    '''mainloop ive used for testing the expansion technique. This uses a
    combinations of bluirring, erosion, a mask, and gradient to get a likely
    pupil out of an image'''
    history = []

    img = cv2.imread('../EyePictures/' + filename,0)
    img = cv2.medianBlur(img,5)
    kernel = np.ones((3,3),np.uint8)

    gray_filtered = cv2.inRange(img, 0, 50)
    # cv2.imshow('detected circles',gray_filtered)
    # cv2.waitKey(0)
    centerIsland = islandProblem(gray_filtered)
    centerIsland = int(centerIsland[0]),int(centerIsland[1])


    mask = cv2.erode(gray_filtered,kernel,iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,2,100,
                                param1=80,param2=60)
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,centerIsland,i[2],255,1, cv2.LINE_AA)
        # # draw the center of the circle
        cv2.circle(img,centerIsland,2,255,3)
        center = (i[0],i[1])
        break

    left,total,right, edge1,edge2 = expandLateral(filename)
    radius = int(edge1-center[0]) + np.argmax(total)
    cv2.circle(img,centerIsland,radius,255,1, cv2.LINE_AA)
    # print center
    # print centerIsland
    # values = getdistanceNumpy(center,img.shape[:2])
    # for i in range(0, width*40,width):
    #     pixels = getNextBand(width, i, values)
    #     history.append(np.mean([img[pix[1]] for pix in pixels]))
        # for val,place in pixels:
        #     img[place] = 255
        # cv2.imshow('detected circles',img)
        # cv2.waitKey(0)
    # plt.plot(history)#this will plot the average pixel value with each expansion
    # history = []
    # plt.show()
    cv2.imshow('detected circles',img)
    cv2.waitKey(0)
#
#
# cv2.destroyAllWindows()

if __name__ == "__main__":
    directory  = '../EyePictures/'

    numEyes = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            displayImg(filename)
        numEyes +=1
        if numEyes > 10: break
