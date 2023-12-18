import cv2 as cv
import time

face_cascade=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade=cv.CascadeClassifier("smile.xml")

video=cv.VideoCapture(0)

while True:
    check,frame=video.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for (x,y,w,h) in face:
        img =cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        smile=smile_cascade.detectMultiScale(gray,scaleFactor=1.8,minNeighbors=20)
        for x,y,w,h in smile:
            img=cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)


    cv.imshow('this',frame)
    key=cv.waitKey(1)

    if key==ord('q'):
        break

video.release()
cv.destroyAllWindows