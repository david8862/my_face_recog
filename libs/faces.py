#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, face_recognition, logging
from threading import Timer, Lock
import numpy as np
import cv2
#from .utils import image_files_in_folder, LOG_FORMAT

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png|gif|bmp)', f, flags=re.I)]

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

class Face_Recognition:
    distance_threshold=0.45
    num_jitters=3

    def __init__(self):
        self.known_names = []
        self.known_face_encodings = []
        self.hist_lock = Lock()
        self.hist_locations = []
        self.hist_face_names = []
        self.hist_distances = []
        self.__init_history_clean_timer()


    def __init_history_clean_timer(self, inc=10):
        #clean the history every 10 seconds
        Timer(inc, self.__clean_history, (inc,)).start()


    def __clean_history(self, inc):
        Timer(inc, self.__clean_history, (inc,)).start()
        if len(self.hist_face_names):
            logging.debug("clean face history")
            with self.hist_lock:
                # clean history result
                self.hist_locations = []
                self.hist_face_names = []
                self.hist_distances = []

    def __same_person(self, previous_location, new_location):
        (top, right, bottom, left) = previous_location
        (_top, _right, _bottom, _left) = previous_location
        if (abs(top-_top)<=20 and abs(right-_right)<=20 and abs(bottom-_bottom)<=20 and  abs(left-_left)<=20):
            return True
        return False


    def scan_known_people(self, known_people_folder, model=None):
        for file in image_files_in_folder(known_people_folder):
            basename = os.path.splitext(os.path.basename(file))[0]
            image = face_recognition.load_image_file(file)
            face_locations = face_recognition.face_locations(image, model='hog')
            #face_locations = face_recognition.face_locations(image, model='cnn')
            encodings = face_recognition.face_encodings(image, face_locations, num_jitters=self.num_jitters)

            if len(encodings) > 1:
                print("WARNING: More than one face found in {}. Only considering the first face.".format(file))
            if len(encodings) == 0:
                print("WARNING: No faces found in {}. Ignoring file.".format(file))
            else:
                self.known_names.append(basename)
                self.known_face_encodings.append(encodings[0])


    def recognize_faces_in_image(self, image_stream, is_file=True, model='hog'):
        logging.debug("step1: Load the uploaded image file")
        if is_file:
            image = face_recognition.load_image_file(image_stream)
        else:
            image = image_stream
        #logging.debug("step1.1: Resize image to 1/4 size for faster face recognition processing")
        #image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        logging.debug("step2: Find all the faces ")
        face_locations = face_recognition.face_locations(image, model=model)
        #face_locations = face_recognition.face_locations(image, model='cnn')
        logging.debug("step3: Get face encodings ")
        face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=self.num_jitters)
        logging.debug("step4: Get face encodings...done")

        face_names = []
        face_distances = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            result = list(distances <= self.distance_threshold)
            index = np.argmin(distances)
            name = "  "
            distance = min(distances)
            # If a match was found in self.known_face_encodings, just use the first one.
            if True in result:
                name = self.known_names[index]

            face_names.append(name)
            face_distances.append(distance)


        with self.hist_lock:
            # save a list of true or  false to indicate the object in PRE_LOCATIONS and in face_locations is the same object
            same_object = []
            if (len(face_locations) == len(self.hist_locations)):
                for location,_location in zip(face_locations, self.hist_locations):
                    same_object.append(self.__same_person(location,_location))

            # After get the result, then compare them with history
            for i in range(len(face_names)):
                if(len(same_object) > 0 and same_object[i] and distance > self.hist_distances[i]):
                    logging.debug("===Use history result====")
                    logging.debug("Previous names list:{}".format(self.hist_face_names))
                    logging.debug("Current face list:{}".format(face_names))
                    face_names[i] = self.hist_face_names[i]
                    logging.debug("Previous face distance:{}".format(self.hist_distances))
                    logging.debug("Current face distance:{}".format(face_distances))
                    face_distances[i] = self.hist_distances[i]
                    logging.debug("After use history, current names:{}, current distance:{}".format(face_names,face_distances))
            # save history result
            self.hist_face_names = face_names
            self.hist_locations = face_locations
            self.hist_distances = face_distances


        face_found = False
        if len(face_encodings) > 0:
            face_found = True

        result = {
            "face_found_in_image": face_found,
            "face_data": {},
        }

        for i, ((top, right, bottom, left), name) in enumerate(zip(face_locations, face_names)):
            face = {
                "name": name,
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left,
                    }
            result['face_data']['face{}'.format(i+1)] = face

        logging.debug("step5: Return the result as json")
        return result


import argparse
import sys
import time
import cv2
def add_overlays(frame, faces, resize_rate):
    if faces is not None:
        for face in faces:
            face_bb = [face['top']*resize_rate, face['right']*resize_rate, face['bottom']*resize_rate, face['left']*resize_rate]
            cv2.rectangle(frame,
                          (face_bb[3], face_bb[0]), (face_bb[1], face_bb[2]),
                          (0, 0, 255), 2)
            if face['name'] is not None:
                # Draw a label with a name below the face
                cv2.rectangle(frame, (face_bb[3], face_bb[2] - 35), (face_bb[1], face_bb[2]), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, face['name'], (face_bb[3] + 6, face_bb[2] - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

def main(args):
    frame_interval = 1  # Number of frames after which to run face detection
    fps_display_interval = 3  # seconds
    frame_rate = 0
    frame_count = 0
    resize_rate = 0.25

    video_capture = cv2.VideoCapture(0)
    recognition = Face_Recognition()
    recognition.scan_known_people(os.path.dirname(os.path.abspath(__file__)) + "/../face_db")
    start_time = time.time()

    if args.debug:
        print("Debug enabled")
        face.debug = True

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            # Resize frame of video for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=resize_rate, fy=resize_rate)
            result = recognition.recognize_faces_in_image(small_frame, is_file=False)
            faces = list(result['face_data'].values())

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, int(1.0/resize_rate))
        frame_count += 1
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    os._exit(0)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
