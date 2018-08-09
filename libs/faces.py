#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, os, face_recognition, logging
from threading import Timer, Lock
import numpy as np
import cv2
from .utils import image_files_in_folder, LOG_FORMAT

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


    def recognize_faces_in_image(self, file_stream, model='hog'):
        logging.debug("step1: Load the uploaded image file")
        image = face_recognition.load_image_file(file_stream)
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

        i = 1
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            face = {
                "name": name,
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left,
                    }
            result['face_data']['face{}'.format(i)] = face
            i = i + 1

        logging.debug("step5: Return the result as json")
        return result

