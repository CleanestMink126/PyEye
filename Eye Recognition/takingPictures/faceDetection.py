import numpy as np
import cv2
import time
import datetime
"""
This code will be the main script for face detection, eye detection, and
door access for the iris recognition Software. First we load the face and eye
classifier so that it can look at images.
"""


face_cascade = cv2.CascadeClassifier('/home/gsteelman/Desktop/Summer Research/opencv/data/harcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/gsteelman/Desktop/Summer Research/opencv/data/harcascades/haarcascade_eye.xml')
success = face_cascade.load('haarcascade_frontalface_default.xml')
success2 = eye_cascade.load('haarcascade_eye.xml')
"""Now we open the videocamera to take in images"""
cap = cv2.VideoCapture(0)
print cap
if not cap:
    ans = cap.open()
    print ans

"""Define parameters"""
timeToNextPicture = 3 #seconds
timeWaitForAnswers = 30
lastTime = time.time()
savePictureBool = True
showVideo = True
waitForAnswers = False
getAnswer = False
sleepLength = 2


"""Main Loop, exit with q"""
while True:
    """Get an image from the video object"""
    ret, frame = cap.read()
    if ret:
        """If video successfully captured, convert to gray and then run face
        and eye detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        eyes = eye_cascade.detectMultiScale(gray,minNeighbors = 25,minSize = (100,100))
        # Display the resulting frame
        """If we detect a face, tell user to get closer"""
        for (x,y,w,h) in faces:
            '''Tell to get closer'''
            print 'closer'
            if showVideo:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        """If we are not waiting a response of the last picture and we see eyes
        get the boxes and if we want to, show the eye picture. Then save the
        eye picture if it is long enough since the last one and update the
        picture history text doument"""
        if eyes != () and not getAnswer:
            (ex,ey,ew,eh) = eyes[0]
            if showVideo:
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                cv2.imshow('frame',frame[ey:ey+eh, ex:ex+ew])
            if savePictureBool:
                cD = str(datetime.datetime.now()).replace(" ","-")[:-7]
                filePath = "EyePictures/"+ cD+".jpg"
                cv2.imwrite( filePath, frame[ey:ey+eh, ex:ex+ew])
                f = open('pictureHistory.txt','a')
                f.write('\n' + cD + '---')
                f.close()
                savePictureBool = False
                if waitForAnswers:
                    getAnswer = True
            elif time.time() - lastTime > timeToNextPicture:
                savePictureBool = True
                lastTime = time.time()
        elif getAnswer:
            '''Add in something to execute the desired matlab scripts
            This part should determine if the last line has been updated
            by looking for --- at the end of the line'''
            picHist = open('pictureHistory.txt','r')
            histLines = picHist.readlines()
            lastPos = len(histLines)
            while getAnswer:
                picHist = open('pictureHistory.txt','r')
                lastLine = picHist.readlines()[-1]
                print lastLine
                print lastLine[-3:]
                if lastLine[-3:] != '---':
                    '''This should determine access and execute the proper commands'''
                    getAnswer = False
                time.sleep(sleepLength)
        elif showVideo:
            cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print 'after loop'
cv2.waitKey(0)
cv2.destroyAllWindows()
