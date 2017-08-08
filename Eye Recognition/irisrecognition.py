import numpy as np
import cv2
import time
import datetime
"""
This code will be the main script for face detection, eye detection, and
door access for the iris recognition Software. First we load the face and eye
classifier so that it can look at images.
"""

"""Now we open the videocamera to take in images"""
cap = cv2.VideoCapture(1)
print cap
if not cap:
    ans = cap.open()
    print ans

"""Define parameters"""
takePicture = False #seconds
timeWaitForAnswers = 5
lastTime = time.time()
savePictureBool = True
showVideo = True
waitForAnswers = True
getAnswer = False
sleepLength = 2

"""Main Loop, exit with q"""
while True:
    """Get an image from the video object"""
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord(' '):
        takePicture = True
    if showVideo and ret:
        cv2.imshow('frame',frame)

    if ret and takePicture:
        takePicture = False
        """If video successfully captured, convert to gray and then run face
        and eye detection"""
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        """If we are not waiting a response of the last picture and we see eyes
        get the boxes and if we want to, show the eye picture. Then save the
        eye picture if it is long enough since the last one and update the
        picture history text doument"""

        if savePictureBool and not getAnswer:
            cD = str(datetime.datetime.now()).replace(" ","-")[:-7]
            filePath = "EyePictures/"+ cD+".jpg"
            cv2.imwrite( filePath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            f = open('pictureHistory.txt','a')
            f.write('\n' + cD + '---')
            f.close()
            if waitForAnswers:
                getAnswer = True

    if getAnswer:
        '''Add in something to execute the desired matlab scripts
        This part should determine if the last line has been updated
        by looking for --- at the end of the line'''
        picHist = open('pictureHistory.txt','r')
        histLines = picHist.readlines()
        lastPos = len(histLines)
        lastTime = time.time()
        cv2.imshow('img',frame)
        while getAnswer:

            picHist = open('pictureHistory.txt','r')
            lastLine = picHist.readlines()[-1]
            print lastLine
            print lastLine[-3:]
            if lastLine[-3:] != '---':
                '''This should determine access and execute the proper commands'''
                getAnswer = False
            time.sleep(sleepLength)
            if time.time() - lastTime > timeWaitForAnswers:
                getAnswer = False
                break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print 'after loop'
cv2.waitKey(0)
cv2.destroyAllWindows()
