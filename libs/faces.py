#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, face_recognition, logging
import numpy as np
import cv2
from .utils import image_files_in_folder, LOG_FORMAT

PRE_LOCATIONS = []
PRE_FACE_NAMES = []
PRE_DISTANCE = []

def same_person(previous_location, new_location):
    (top, right, bottom, left) = previous_location
    (_top, _right, _bottom, _left) = previous_location
    if (abs(top-_top)<=20 and abs(right-_right)<=20 and abs(bottom-_bottom)<=20 and  abs(left-_left)<=20):
        return True
    return False


logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
DISTANCE_THRESHOLD=0.45
NUM_JITTERS=3

def init_model():
    return None

def scan_known_people(known_people_folder, model=None):
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        image = face_recognition.load_image_file(file)
        face_locations = face_recognition.face_locations(image, model='hog')
        #face_locations = face_recognition.face_locations(image, model='cnn')
        encodings = face_recognition.face_encodings(image, face_locations, num_jitters=NUM_JITTERS)

        if len(encodings) > 1:
            print("WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            print("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])

    return known_names, known_face_encodings


def recognize_faces_in_image(file_stream, known_face_names, known_face_encodings, model='hog'):
    logging.debug("step1: Load the uploaded image file")
    image = face_recognition.load_image_file(file_stream)
    logging.debug("step2: Find all the faces ")
    face_locations = face_recognition.face_locations(image, model=model)
    #face_locations = face_recognition.face_locations(image, model='cnn')
    logging.debug("step3: Get face encodings ")
    face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=NUM_JITTERS)
    logging.debug("step4: Get face encodings...done")

    global PRE_DISTANCE
    global PRE_FACE_NAMES
    global PRE_LOCATIONS

    # save a list of true or  false to indicate the object in PRE_LOCATIONS and in face_locations is the same object
    same_object = []
    if (len(face_locations) == len(PRE_LOCATIONS)):
        for location,_location in zip(face_locations, PRE_LOCATIONS):
            same_object.append(same_person(location,_location))


    face_names = []
    face_distances = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        result = list(distances <= DISTANCE_THRESHOLD)
        index = np.argmin(distances)
        name = "  "
        distance = min(distances)
        # If a match was found in known_face_encodings, just use the first one.
        if True in result:
            name = known_face_names[index]

        face_names.append(name)
        face_distances.append(distance)

    # After get the result, then compare them with history
    for i in range(len(face_names)):
        if(len(same_object) > 0 and same_object[i] and distance > PRE_DISTANCE[i]):
            logging.debug("===Use history result====")
            logging.debug("Previous names list:{}".format(PRE_FACE_NAMES))
            logging.debug("Current face list:{}".format(face_names))
            face_names[i] = PRE_FACE_NAMES[i]
            logging.debug("Previous face distance:{}".format(PRE_DISTANCE))
            logging.debug("Current face distance:{}".format(face_distances))
            face_distances[i] = PRE_DISTANCE[i]
            logging.debug("After use history, current names:{}, current distance:{}".format(face_names,face_distances))
    # save history result
    #PRE_FACE_NAMES = face_names
    #PRE_LOCATIONS = face_locations
    #PRE_DISTANCE = face_distances

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


def recognize_faces_in_image_fast(file_stream, known_face_names, known_face_encodings, model='hog'):
    logging.debug("step1: Load the uploaded image file")
    image = face_recognition.load_image_file(file_stream)

    logging.debug("step2: Resize image to 1/4 size for faster face recognition processing")
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    logging.debug("step3: Find all the faces and face encodings in the current frame of video")
    face_locations = face_recognition.face_locations(small_image, model=model)
    #face_locations = face_recognition.face_locations(small_image, model='cnn')

    logging.debug("step4: Get face encodings for any faces in the uploaded image")
    face_encodings = face_recognition.face_encodings(small_image, face_locations, num_jitters=NUM_JITTERS)


    logging.debug("step5: find out all face_names")
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        result = list(distances <= 0.45)
        index = np.argmin(distances)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in result:
            name = known_face_names[index]

        face_names.append(name)

    face_found = False
    if len(face_encodings) > 0:
        face_found = True

    result = {
        "face_found_in_image": face_found,
        "face_data": {},
    }

    i = 1
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        face = {
            "name": name,
            "top": top,
            "right": right,
            "bottom": bottom,
            "left": left,
                }
        result['face_data']['face{}'.format(i)] = face
        i = i + 1

    logging.debug("step6: Return the result as json")
    return result

