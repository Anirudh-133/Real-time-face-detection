import cv2
from random import randrange
#using trained data
trained_face_data =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img=cv2.imread('rdj4.PNG')


#img to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect  faces
face_coordinates= trained_face_data.detectMultiScale(grayscaled_img)


#Draw rectangle around the face
for (x,y,w,h) in face_coordinates:

 cv2.rectangle(img,( x ,y) , (x+ w, y+h),(randrange(256),randrange(256),randrange(256)),10)

cv2.imshow('jack',img)
cv2.waitKey() #so we can see the img
    

#to close when pressed q



