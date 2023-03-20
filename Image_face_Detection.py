# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 20:41:18 2023

@author: hp
"""

import face_recognition
import cv2

#loading the image to detect
image_detect = cv2.imread('images/people.jpg')
cv2.imshow('test',image_detect)

#detect all faces in the image
all_face_locations = face_recognition.face_locations(image_detect,model='hog')

#print the number of faces detected
print('There are {} no of faces in the image  '.format(len(all_face_locations)))

#looping through the face locations
for index,cur_face_loc in enumerate(all_face_locations):
    #splitting the tuple to get the four position values of current face
    top,right,bottom,left = cur_face_loc
    #printing the location of current face
    #print('Found faces {} at the top :{} , right :{}, bottom:{}, left:{}'.format(index+1,top,right,bottom,left))
    
    cv2.rectangle(all_face_locations,(left,top),(right,bottom),(0,255,0),2)
    #slicing the current face from main image
    cur_face_image = image_detect[top : bottom ,left : right]
    #showing the current face with dynamic title
    #cv2.imshow("face no: "+ str(index+1),cur_face_image)