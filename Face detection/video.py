import cv2
from random import randrange


#using trained data
trained_face_data =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#to capture video
webcam=cv2.VideoCapture(0)   #give 0 to read from webcam or give the location for file

while True:
    successful_frame_read,frame =webcam.read()
     
    #convert frame to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detect faces
    face_coordinates= trained_face_data.detectMultiScale(grayscaled_img)
    
    #Draw rectangle around the face
    for (x,y,w,h) in face_coordinates:

          cv2.rectangle(frame,( x ,y) , (x+ w, y+h),(randrange(256),randrange(256),randrange(256)),10)
    
    cv2.imshow('jack',frame)
    key=cv2.waitKey(1) #here now we put 1 so we can see frame by frame

    #to close when pressed q
    if key==81 or key==113:
          break






