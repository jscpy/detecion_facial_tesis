# -*- coding: utf-8 -*-
import cv2 as cv
from imutils import paths
# face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
human_cascade = cv.CascadeClassifier('cascades/haarcascade_fullbody.xml')

for imagePath in paths.list_images('lab'):
    img = cv.imread(imagePath)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    persons = human_cascade.detectMultiScale(gray)
    # faces = face_cascade.detectMultiScale(gray, 1.10, 4)
    for (x, y, w, h) in persons:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print(imagePath)
    # roi_gray = gray[y:y+h, x:x+w]
    # roi_color = img[y:y+h, x:x+w]
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
    #    cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
