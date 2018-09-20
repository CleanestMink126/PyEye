import cv2
import numpy as np
import os
from operator import itemgetter
import Queue
from pupilDeps import *

width = 3
bins = np.arange(256)
def irisLikelihood(myImg, numBoxes):
    '''This function will divide the image into a set number of boxes, Then
    for each box it will compare the histogram of the box to that of the confirmed
    Iris to come up with a likelyhood of that being iris'''
    img = np.array(myImg.img)
    xb,yb, =myImg.xs//numBoxes,myImg.ys//numBoxes
    numx,numy= myImg.xs/xb,myImg.ys/yb
    size = xb*yb
    for i in range(numx):
        for j in range(numy):
            matrix = img[j*yb:(j+1)*yb,i*xb:(i+1)*xb]
            hist = np.histogram(matrix,bins=bins)[0]
            # print(hist)
            hist = np.array(hist,dtype=np.float32)/size
            # print(hist)
            likelihood = np.sum(np.sqrt(hist * myImg.histogram))
            img[j*yb:(j+1)*yb,i*xb:(i+1)*xb] = likelihood*255
    myImg.likelihood = img

def irisLikelihoodVairable(myImg, size,spacing):
    '''This will do the same thing as iris likelyhood, but have some control over
    the size and spacing of the boxes, in the same style of strided convolutions'''
    if myImg.pupilRad is None:
        getBaseline(myImg)
    if myImg.irisTerritory is None:
        myImg.irisTerritory = myImg.highestBand
    myImg.histogram =  np.array(np.histogram(myImg.irisTerritory,bins = bins)[0],np.float32)/len(myImg.irisTerritory)
    img = np.array(myImg.img)
    holder = np.zeros(img.shape,dtype=np.float32)
    area= size[1] * size[0]
    numx,numy= 1 + (myImg.xs-size[0])/spacing[0],1 + (myImg.ys-size[1])/spacing[1]
    for i in range(numx):
        for j in range(numy):
            matrix = img[j*spacing[1]:j*spacing[1] + size[1] ,i*spacing[0]:i*spacing[0] + size[0]]
            hist = np.histogram(matrix,bins=bins)[0]
            hist = np.array(hist,dtype=np.float32)/area
            # print(sum(hist))
            likelihood = 255.0* np.sum(np.sqrt(hist * myImg.histogram))
            # print(likelihood)
            holder[size[1]//2 + j*spacing[1]:size[1]//2 +(j+1)*spacing[1] ,size[0]//2 + i*spacing[0]:size[0]//2 + (i+1)*spacing[0]] = likelihood
    img[:,:] = holder
    myImg.likelihood =  img

def calculateDiff(myImg):
    myImg.diff = np.diff(myImg.img.astype(np.float32), axis = 0).astype(np.uint8)

def examineLikelihood(myImg,numBoxes):
    '''get and set the likelihood of being an iris of the pixels'''
    if myImg.pupilRad is None:
        getBaseline(myImg)
    if myImg.irisTerritory is None:
        myImg.irisTerritory = myImg.highestBand
    myImg.histogram =  np.array(np.histogram(myImg.irisTerritory,bins = bins)[0],np.float32)/len(myImg.irisTerritory)
    irisLikelihood(myImg,numBoxes)


def getMask(myImg):
    '''Get a mask to put over the image to see where the iris is segmented'''
    x = np.arange(myImg.xs)
    y = np.transpose(np.arange(myImg.ys))
    z = np.sqrt((x-myImg.center[0])**2 + (y[:, np.newaxis]-myImg.center[1])**2)
    return (myImg.pupilRad<z) & (z < myImg.irisRad)

def checkAlterantive(mask):
    '''Find circles using HoughCircles transform'''
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,2,100,
                                param1=80,param2=60)
    return circles

def displayLikelihood(myImg):
    irisLikelihoodVairable(myImg,size=[10,10] ,spacing=[2,2])
    cv2.imshow('total likelihood',myImg.likelihood)
    cv2.waitKey(0)
    myImg.likelihood[:,:] = np.where(getMask(myImg),myImg.likelihood[:,:],0)
    myImg.img[:,:] = np.where(getMask(myImg),myImg.img[:,:],0)
    cv2.imshow('cut image',myImg.img)
    cv2.waitKey(0)
    cv2.imshow('cut likelihood',myImg.likelihood)
    cv2.waitKey(0)

def setViablePixels(myImg, threshold):
    myImg.viablePixels = np.ones(myImg.img.shape) * 255
    myImg.viablePixels[myImg.likelihood<threshold] = 0

def getCircles(filename):
    '''Main Loop used by other programs to dissect an image'''
    history = []
    myImg = imageContainer(filename)
    myImg.center = myImg.xs,myImg.ys
    img = cv2.medianBlur(myImg.img,5)
    kernel = np.ones((3,3),np.uint8)
    gray_filtered = cv2.inRange(img, 0, 60)
    mask = cv2.erode(gray_filtered,kernel,iterations = 1)
    # print("masks done")
    centerIsland = islandProblem(mask,myImg)
    myImg.center = int(centerIsland[0]),int(centerIsland[1])
    # print("island done")
    circles = checkAlterantive(mask)
    left,total,right = expandLateral(myImg)
    # print("found edges")
    myImg.irisRad = int(myImg.edgeRight-myImg.center[0]) + np.argmax(total)
    if circles is not None:
        for i in circles[0,:]:
            # draw the outer circle
            myImg.pupilRad+= i[2]
            myImg.pupilRad =int(myImg.pupilRad//2)
            # print("adjusted pupilRad")
            break

    examineLikelihood(myImg, 100)
    # calculateDiff(myImg)
    setViablePixels(myImg, 30)
    # cv2.circle(myImg.img,myImg.center,2,255,3)
    # cv2.circle(myImg.img,myImg.center,int(myImg.pupilRad),255,1, cv2.LINE_AA)
    # cv2.circle(myImg.img,myImg.center,int(myImg.irisRad),255,1, cv2.LINE_AA)
    return myImg

def displayImg(filename):
    '''mainloop ive used for testing the expansion technique. This uses a
    combinations of bluirring, erosion, a mask, and gradient to get a likely
    pupil out of an image'''
    history = []
    myImg = imageContainer(filename)
    myImg.center = myImg.xs,myImg.ys
    cv2.imshow('detected circles',myImg.img)
    cv2.waitKey(0)

    img = cv2.medianBlur(myImg.img,5)
    kernel = np.ones((3,3),np.uint8)
    gray_filtered = cv2.inRange(img, 0, 60)
    mask = cv2.erode(gray_filtered,kernel,iterations = 1)
    print("masks done")

    centerIsland = islandProblem(mask,myImg)
    myImg.center = int(centerIsland[0]),int(centerIsland[1])
    print("island done")

    circles = checkAlterantive(mask)
    left,total,right = expandLateral(myImg)
    print("found edges")

    myImg.irisRad = int(myImg.edgeRight-myImg.center[0]) + np.argmax(total)
    if circles is not None:
        for i in circles[0,:]:
            # draw the outer circle
            myImg.pupilRad+= i[2]
            myImg.pupilRad =int(myImg.pupilRad//2)
            print("adjusted pupilRad")
            break

    cv2.circle(myImg.img,myImg.center,2,255,3)
    cv2.circle(myImg.img,myImg.center,int(myImg.pupilRad),255,1, cv2.LINE_AA)
    cv2.circle(myImg.img,myImg.center,int(myImg.irisRad),255,1, cv2.LINE_AA)

    displayLikelihood(myImg)
    return myImg

if __name__ == "__main__":
    chance = 0.05
    directory  = '../EyePictures/'

    numEyes = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") and np.random.rand() < chance:
            displayImg(filename)
            numEyes +=1
        if numEyes > 10: break
