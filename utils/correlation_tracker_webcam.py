#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This example shows how to use the correlation_tracker from the dlib Python
# library to track a face. To use it, you use face_recognition lib to capture the
# bounding box of people's face from live video of your webcam. Then it will
# identify the location of the face in subsequent frames.
#
# actually, you can use the correlation_tracker to track any moving object


import face_recognition
import cv2
import dlib

def face_track():
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    tracker = dlib.correlation_tracker()
    k = 0

    while True:
        # Grab a single frame of video
        _, frame = video_capture.read()

        # We need to initialize the tracker on the first frame
        if k == 0:
            face_locations = face_recognition.face_locations(frame, model='hog')
            if len(face_locations):
                face_location = face_locations[0]
                # Start a track on the juice box. If you look at the first frame you
                # will see that the juice box is contained within the bounding
                # box (74, 67, 112, 153).
                tracker.start_track(frame, dlib.rectangle(left=face_location[3], top=face_location[0], right=face_location[1], bottom=face_location[2]))
                k = k + 1
        else:
            # Else we just attempt to track from the previous frame
            tracker.update(frame)
            rect = tracker.get_position()
            cv2.rectangle(frame, (int(rect.left()), int(rect.top())), (int(rect.right()), int(rect.bottom())), (0, 0, 255), 2)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    face_track()


if __name__ == "__main__":
    main()
