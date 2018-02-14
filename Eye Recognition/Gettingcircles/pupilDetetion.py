import cv2
import numpy as np
import os
from operator import itemgetter
import heapq
import matplotlib.pyplot as plt
import copy
import Queue

# def getBorderPixels(image, center, radius, thickness):
#     '''The idea of this function is to return a set of tuples representing x,y
#     coordinates of pixels along the circuference of a given circle'''
#     pass
# def getDataFromMask(mask):
#     '''This might find the pupil given the mask'''

def getdistanceNumpy(center, size):
    '''this function uses numpy to generate a matrix whose values are distances
    from the center of the proposed circle. the values of this circle are then
    taken out along witht he indices and put into a min heap'''
    x = np.arange(size[1])
    y = np.transpose(np.arange(size[0]))
    z = (x-center[0])**2 + (y[:, np.newaxis]-center[1])**2
    values = [(value, index)for index, value in np.ndenumerate(z)]
    heapq.heapify(values)
    return values

def getNextBand(width, radius, heap):
    '''this function will return an appropriate amount of correct pixels given the
    current radius of a circle and the width of pixels to examine from there.
    It just pops from the heap created above'''
    numBoxes = round(np.pi * ((2* radius * width) + width**2))
    # print(numBoxes)
    return [heapq.heappop(heap) for i in range(int(numBoxes))]

# values = getdistanceNumpy((100,100),(240,480))
width = 3
# for i in range(0, width*20,width):
#     print getNextBand(width, i, values)
def getMaxHeap(filename,blur = 5):
    '''quickly get the heap of pixels from the image'''

    img = cv2.imread('../EyePictures/' + filename,0)
    img2 = cv2.medianBlur(img,blur)
    kernel = np.ones((3,3),np.uint8)



    gray_filtered = cv2.inRange(img2, 0, 60)
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

    return getdistanceNumpy(center,img.shape[:2]), img, center


def getBaseline(filename, cutoff= 2):
    '''the goal of this function is to find the brightest band in the iris and use
    that to inform about whether or not an individual pixel belongs to the iris'''
    highestBand = []
    highestMean = 0
    history = [0]#this stores the previous 5 values, is a queue
    dangaZone = False
    '''we don't want to start looking for a max until we are in
    iris territory so this will approximate when we reach that point'''
    values, img, center = getMaxHeap(filename)
    values2 = copy.deepcopy(values)
    for i in range(0, width*40,width):
        pixels = getNextBand(width, i, values)#get the pixels to be indexed in the image
        band = [img[pix[1]] for pix in pixels]#get the values
        mean = np.mean(band)
        # print mean
        hstmean = np.mean(history)
        if (not dangaZone) and len(history)>3:
            std = np.std(history)# this part is saying if the current mean is
            #significantly above the past means (i.e. stuff went from black to less
            #black)
            if std != 0 and (mean - hstmean)/std > cutoff:
                dangaZone = True
                pupilRad = i

        if dangaZone and mean > highestMean:
            #after we know we are in iris territory, keep track of highest mean band
            highestMean = mean
            highestBand = band
        elif dangaZone and mean < hstmean:
            #if we get to the point where we are no longer increasing, return
            return highestBand,i, pupilRad, values2, img, center

        history.append(mean)
        if len(history)>5:
            history.pop(0)

    print "never returned"
    print dangaZone


# def changeValue(img, ):

def examineBaselinePlot(filename):
    highestBand,radius, pupilRad, values, img, center = getBaseline(filename)
    mean = np.mean(highestBand)
    std = np.std(highestBand)
    history = []
    for i in range(0, width*40,width):
        pixels = getNextBand(width, i, values)
        band =[img[pix[1]] for pix in pixels]
        band = abs(band - mean)/std
        history.append(np.mean(band))

    plt.plot(history)#this will plot the average pixel value with each expansion
    history = []
    plt.show()

def examineBaseline(filename):
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


def walkOneSide(direction, radius, img,center,historySize = 10,incrementSTD = 10, cutoff = 2.5):
    history = []#this really should be a LL or QUEUE but makes np.mean hard
    edge = center[0]+direction*radius
    for i in range(historySize):
        # print (edge - direction * i, center[1])
        history.append(img[(int(center[1]),int(edge - direction * i))])
    historyMean = np.mean(history)
    std = np.std(history)

    for i in range(0,int(min(center[0], img.shape[1] - center[0])- radius)):
        if not (i % incrementSTD):
            std = np.std(history)
        print history, "1"
        print historyMean
        new = float(img[(int(center[1]),int(edge + direction * i))])
        # print new
        if (new - historyMean)/std > cutoff:
            return radius + i
        old = float(history.pop(0))
        history.append(new)
        diff =float(new - old)
        historyMean += diff / historySize

def walkOneSideBetter(direction, radius, img,center,historySize = 10,incrementSTD = 1, cutoff = 0.5):
    data = []
    history = []#this really should be a LL or QUEUE but makes np.mean hard
    future = []
    edge = center[0]+direction*radius
    for i in range(historySize):
        # print (edge - direction * i, center[1])
        history.append(float(img[(int(center[1]),int(edge - direction * i))]))
        future.append(float(img[(int(center[1]),int(edge + direction * i))]))

    historyMean = np.mean(history)
    stdH = np.std(history)

    futureMean = np.mean(future)
    stdF = np.std(future)

    for i in range(0,int(min(center[0], img.shape[1] - center[0])- radius- historySize)):
        if not (i % incrementSTD):
            stdH = np.std(history)
            stdF = np.std(future)
        newF = float(img[(int(center[1]),int(edge + direction * (i+ historySize)))])
        newH = future.pop(0)
        oldH = history.pop(0)
        history.append(newH)
        future.append(newF)
        diff = newF - newH
        futureMean += diff / historySize
        diff = newH - oldH
        historyMean += diff / historySize
        if futureMean < historyMean:
            d = 0
        else:
            d = (futureMean-historyMean)/np.hypot(stdH,stdF)
        data.append(d)


    return 255 * np.array(data)/max(data), edge

def expandLateral(filename):
    _,radius, _, values, img, center = getBaseline(filename)
    irisRad1,edge1 = walkOneSideBetter(1,radius,img,center)
    irisRad2,edge2 = walkOneSideBetter(-1,radius,img,center)
    print len(irisRad1), edge1
    print len(irisRad2), edge2
    total = irisRad1 * irisRad2
    print total
    total = 255 * total/max(total)
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
    plt.plot(history)#this will plot the average pixel value with each expansion
    history = []
    plt.show()
    cv2.imshow('detected circles',img)
    cv2.waitKey(0)


        # print(os.path.join(directory, filename))
#
#
# cv2.destroyAllWindows()

if __name__ == "__main__":
    directory  = '../EyePictures/'

    numEyes = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            expandLateral(filename)
        numEyes +=1
        if numEyes > 3: break
