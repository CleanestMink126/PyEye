import heapq
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2
# def getBorderPixels(image, center, radius, thickness):
#     '''The idea of this function is to return a set of tuples representing x,y
#     coordinates of pixels along the circuference of a given circle'''
#     passs
# def getDataFromMask(mask):
#     '''This might find the pupil given the mask'''
class BaselineError():
    pass

class imageContainer:
    def __init__(self,filepath):
        self.img = cv2.imread(filepath,0)
        self.ys,self.xs = self.img.shape[:2]
        self.width =1
        self.pupilRad = None
        self.valueHeap = None
        self.irisRad = None
        self.center = None
        self.mask = None
        self.maxRad = None
        self.highestBand = None
        self.irisTerritory = None
        self.histogram = None
        self.likelihood=None


def createMaxHeap(myImg,blur = 5):
    '''quickly get the heap of pixels from the image'''
    if myImg.center is None:
        img2 = cv2.medianBlur(myImg.img,blur)
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
        myImg.center = center
    getdistanceNumpy(myImg)

def getdistanceNumpy(myImg):
    '''this function uses numpy to generate a matrix whose values are distances
    from the center of the proposed circle. the values of this circle are then
    taken out along witht he indices and put into a min heap'''
    x = np.arange(myImg.xs)
    y = np.transpose(np.arange(myImg.ys))
    z = np.sqrt((x-myImg.center[0])**2 + (y[:, np.newaxis]-myImg.center[1])**2)
    values = [(value, index)for index, value in np.ndenumerate(z)]
    heapq.heapify(values)
    myImg.valueHeap = values


def walkOneSide(direction, myImg,historySize = 10,incrementSTD = 10, cutoff = 2.5):
    '''After getting the baseline a.k.a. the brightest part of the iris, this method
    will walk outwards horizontally toward the sclera, comparing the values before
     current point with the current point. When the next point is above a certain
     standard deviation away it will return the radius'''
    history = []#this really should be a LL or QUEUE but makes np.mean hard
    edge = myImg.xs+direction*myImg.maxRad
    for i in range(historySize):
        # print (edge - direction * i, center[1])
        history.append(myImg.img[(int(myImg.ys),int(edge - direction * i))])
    historyMean = np.mean(history)
    std = np.std(history)

    for i in range(0,int(min(myImg.xs, img.xs - myImg.xs)- myImg.maxRad)):
        if not (i % incrementSTD):
            std = np.std(history)
        print history, "1"
        print historyMean
        new = float(myImg.img[(int(myImg.ys),int(edge + direction * i))])
        # print new
        if (new - historyMean)/std > cutoff:
            return myImg.maxRad + i
        old = float(history.pop(0))
        history.append(new)
        diff =float(new - old)
        historyMean += diff / historySize

def walkOneSideBetter(direction, myImg,historySize = 10,incrementSTD = 1, cutoff = 0.5):
    '''After getting the baseline a.k.a. the brightest part of the iris, this method
    will walk outwards horizontally toward the sclera, comparing the values before
    and after the current point at which it is looking. It compiles these into a
    lits and returns the list. it compares using a t test varient'''
    data = []
    history = []#this really should be a LL or QUEUE but makes np.mean hard
    future = []
    edge = myImg.center[0]+direction*myImg.maxRad
    for i in range(historySize):
        # print (edge - direction * i, center[1])
        history.append(float(myImg.img[(int(myImg.center[1]),int(edge - direction * i))]))
        future.append(float(myImg.img[(int(myImg.center[1]),int(edge + direction * i))]))

    historyMean = np.mean(history)
    stdH = np.std(history)

    futureMean = np.mean(future)
    stdF = np.std(future)

    for i in range(0,int(min(myImg.center[0], myImg.xs - myImg.center[0])- myImg.maxRad- historySize)):
        if not (i % incrementSTD):
            stdH = np.std(history)
            stdF = np.std(future)
        newF = float(myImg.img[(int(myImg.center[1]),int(edge + direction * (i+ historySize)))])
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
    return np.array(data)/max(data), edge

def getBaseline(myImg,width =3,saveTerritory = True):
    '''the goal of this function is to find the brightest band in the iris and use
    that to inform about whether or not an individual pixel belongs to the iris'''
    irisTerritory = []
    highestBand = []
    highestMean = 0
    history = []
    diffHistory = []
    dangaZone = False
    thres = None
    '''we don't want to start looking for a max until we are in
    iris territory so this will approximate when we reach that point'''
    maxRange = int(min(myImg.xs - myImg.center[0],myImg.center[0],myImg.ys - myImg.center[1],myImg.center[1]))
    getdistanceNumpy(myImg)
    width = myImg.width

    for i in range(0, maxRange-width,width):
        pixels = getNextBand(width, i, myImg.valueHeap)#get the pixels to be indexed in the image
        band = np.array([myImg.img[pix[1]] for pix in pixels])#get the values
        mean = np.mean(band)
        # if i <1:
        #     print(band)
        # print(mean)

        if thres == None or mean < thres:
            thres = mean
        bandmin = np.min(band)
        history.append(mean)
        if (not dangaZone) and len(history)-1:
            diffHistory.append(history[-1]-history[-2])
            if len(diffHistory)-1 and diffHistory[-1]<diffHistory[-2] and bandmin > thres:
                dangaZone = True
                pupilRad = i
        if dangaZone and mean > highestMean:
            #after we know we are in iris territory, keep track of highest mean band
            highestMean = mean
            highestBand = band
            if saveTerritory:
                irisTerritory = np.concatenate((irisTerritory,band))
        elif dangaZone and mean < highestMean:
            #if we get to the point where we are no longer increasing, return
            if saveTerritory:
                irisTerritory = np.concatenate((irisTerritory,band))
                myImg.irisTerritory = irisTerritory
            myImg.histestBand = highestBand
            myImg.pupilRad = pupilRad
            myImg.maxRad = i
            return True

    return False

def getBaselineOld(filename, cutoff= 2,width =3):
    '''DEPRECATED ---------------------------------------------'''
    '''the goal of this function is to find the brightest band in the iris and use
    that to inform about whether or not an individual pixel belongs to the iris'''
    highestBand = []
    highestMean = 0
    history = [0]#this stores the previous 5 values, is a queue
    dangaZone = False
    '''we don't want to start looking for a max until we are in
    iris territory so this will approximate when we reach that point'''
    values, img, center = createMaxHeap(filename)
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

def examineBaselinePlot(filename):
    '''DEPRECATED ----------------------------'''
    '''this will plot the average pixel value as the radius expands from the
    center of the pupil'''
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


def getNextBand(width, radius, heap):
    '''this function will return an appropriate amount of correct pixels given the
    current radius of a circle and the width of pixels to examine from there.
    It just pops from the heap created above'''
    numBoxes = round(np.pi * ((2* radius * width) + width**2))
    # print(numBoxes)
    return [heapq.heappop(heap) for i in range(int(numBoxes))]

def islandProblem(mask,myImg):
    global mean
    global number
    global foundOne
    global start
    mean = [0.0,0.0]
    number = 0
    ys,xs = mask.shape[:2]
    xmid = xs/2
    ymid = ys/2
    getdistanceNumpy(myImg)
    start = 1
    foundOne = 1
    width = 1
    i = 0
    # print(values)
    # print(values[:100])
    def updateMean(newNum,maskVal):
        global mean
        global number
        global foundOne
        global start
        if maskVal:
            number +=1
            mean[0] += (newNum[1][0]-mean[0])/number
            mean[1] += (newNum[1][1]-mean[1])/number
            foundOne = 1
            start = 0

    while start or foundOne:
        foundOne = 0
        pixels = getNextBand(width, i, myImg.valueHeap)
        i += width
        [updateMean(newNum,mask[newNum[1]]) for newNum in pixels]

    mean = mean[::-1]
    return mean


def expandLateral(myImg):
    '''This method will use the better walk to get the lists of one side of the
    eye with the other. It will then multuply the lists together to get an approximate of
    where a radius could be'''
    if not getBaseline(myImg):
        raise BaselineError()
    # print(getBaseline(myImg))
    irisRadRight,edgeRight = walkOneSideBetter(1,myImg)
    irisRadLeft,edgeLeft = walkOneSideBetter(-1,myImg)
    total = irisRadLeft * irisRadRight
    total = 255 * total/max(total)
    myImg.edgeRight = edgeRight
    myImg.edgeLeft = edgeLeft
    return 255*irisRadLeft,total,255*irisRadRight
