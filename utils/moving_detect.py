#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# A simple example of shows how to use the correlation_tracker from the dlib Python
# library to track a face. To use it, you use face_recognition lib to capture the
# bounding box of people's face from live video of your webcam. Then it will
# identify the location of the face in subsequent frames.
#
# actually, you can use the correlation_tracker to track any moving object

import cv2
import numpy as np

camera = cv2.VideoCapture(0)
if (camera.isOpened()):
    print('Open')
else:
    print('Camera not open')

# check video size
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('size:'+repr(size))

es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
background = None
count = 0

while True:
    grabbed, frame_lwpCV = camera.read()
    count = count +1
    # convert frame to gray
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    # Gaussian blur to filter noise in frame, caused by nature shift/light/cam
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    # pick the 30th frame as background, to get ride of the unstable cam bootup
    if background is None:
        if count == 30:
            background = gray_lwpCV
            continue
        else:
            continue
    # check difference of every frame with background to get a different map
    # apply threshold to get black-white
    diff = cv2.absdiff(background, gray_lwpCV)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate to normalize any hole & imperfection
    diff = cv2.dilate(diff, es, iterations=2)

    # get target outline in frame
    image, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 5000: # only pick rectangle large than threshold to filter noise
            continue
        (x, y, w, h) = cv2.boundingRect(c) # get rectangle bounding
        cv2.rectangle(frame_lwpCV, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('dis', diff)
    cv2.imshow('contours', frame_lwpCV)

    key = cv2.waitKey(1) & 0xFF
    # Hit 'q' on the keyboard to quit!
    if key == ord('q'):
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()


