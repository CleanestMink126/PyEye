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
bins = np.arange(256)
def irisLikelihood(myImg, numBoxes):
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
    myImg.likelihood =  img

def irisLikelihoodVairable(myImg, size,spacing):
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
    # myImg.irisMean = np.mean(myImg.irisTerritory)
    # myImg.irisMean = np.mean(myImg.irisTerritory)

# def alternativeLikelihood():
#     myImg.img


def examineLikelihood(myImg,numBoxes):
    '''mainloop ive used for testing the expansion technique. This uses a
    combinations of bluirring, erosion, a mask, and gradient to get a likely
    pupil out of an image'''
    if myImg.pupilRad is None:
        getBaseline(myImg)
    if myImg.irisTerritory is None:
        myImg.irisTerritory = myImg.highestBand
    myImg.histogram =  np.array(np.histogram(myImg.irisTerritory,bins = bins)[0],np.float32)/len(myImg.irisTerritory)
    irisLikelihood(myImg,numBoxes)


def getMask(myImg):
    x = np.arange(myImg.xs)
    y = np.transpose(np.arange(myImg.ys))
    z = np.sqrt((x-myImg.center[0])**2 + (y[:, np.newaxis]-myImg.center[1])**2)
    return (myImg.pupilRad<z) & (z < myImg.irisRad)

#

# def changeValue(img, ):
def examineBaseline(filename):
    '''DEPRECATED -------------------------'''
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
    '''DEPRECATED----------------'''
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

def checkAlterantive(mask):
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,2,100,
                                param1=80,param2=60)
    print("Circles found")
    return circles

def displayLikelihood(myImg):
    irisLikelihoodVairable(myImg,size=[10,10] ,spacing=[2,2])
    print("Examined likelihood")
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
    history = []
    myImg = imageContainer(filename)
    myImg.center = myImg.xs,myImg.ys
    # cv2.imshow('detected circles',myImg.img)
    # cv2.waitKey(0)

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

    examineLikelihood(myImg, 100)
    calculateDiff(myImg)
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

#
#
# cv2.destroyAllWindows()

if __name__ == "__main__":
    chance = 0.05
    directory  = '../EyePictures/'

    numEyes = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") and np.random.rand() < chance:
            displayImg(filename)
            numEyes +=1
        if numEyes > 10: break
