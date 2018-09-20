import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/home/gsteelman/Desktop/Summer Research/opencv/data/harcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/gsteelman/Desktop/Summer Research/opencv/data/harcascades/haarcascade_eye.xml')
success = face_cascade.load('haarcascade_frontalface_default.xml')
success2 = eye_cascade.load('haarcascade_eye.xml')
print success
print 'loaded'
img = cv2.imread('nickFace.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img, 2, 5)

for (x,y,w,h) in faces:
    '''Tell to get closer'''
    print 'closer'
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

eyes = eye_cascade.detectMultiScale(gray)
for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img[ey:ey+eh, ex:ex+ew])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print 'after loop'
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
