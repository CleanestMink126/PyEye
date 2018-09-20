import numpy as np
import cv2
import datetime
cap = cv2.VideoCapture(1)
personName = "Yichen/"
takePicture = False
if not cap:

    ans = cap.open()
    print(ans)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    # Our operations on the frame come here
    if ret:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            takePicture = True
        if takePicture:
            takePicture = False
            cD = str(datetime.datetime.now()).replace(" ","-")[:-7]
            filePath = "./../EyePictures/"+personName+ cD+".jpg"
            cv2.imwrite( filePath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # f = open('pictureHistory.txt','a')
            # f.write('\n' + cD + '---')
            # f.close()
            # if waitForAnswers:
            #     getAnswer = True
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
