import cv2 as cv
import numpy as np

face_cascade_classifier = cv.CascadeClassifier("C:/Users/Ben/Documents/OpenCV/opencv/sources/data/haarcascade_frontalface_default.xml")
if face_cascade_classifier.empty() == True:
    print("No XML")
else:
    img = cv.imread('Ben_Hers.jpg')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = face_cascade_classifier.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv.imshow('img',img)
    cv.waitKey(0)
    cv.destroyAllWindows()