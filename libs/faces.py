#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, face_recognition, logging
import numpy as np
import cv2

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png|gif)', f, flags=re.I)]

def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        image = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 1:
            print("WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            print("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])

    return known_names, known_face_encodings

def allowed_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recognize_faces_in_image(file_stream, known_face_names, known_face_encodings):
    logging.debug("step1: Load the uploaded image file")
    image = face_recognition.load_image_file(file_stream)
    logging.debug("step2: Find all the faces ")
    face_locations = face_recognition.face_locations(image, model='hog')
    logging.debug("step3: Get face encodings ")
    face_encodings = face_recognition.face_encodings(image, face_locations)

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
        face = {
            "name": name,
            "top": top,
            "right": right,
            "bottom": bottom,
            "left": left,
                }
        result['face_data']['face{}'.format(i)] = face
        i = i + 1

    logging.debug("step4: Return the result as json")
    return result


def recognize_faces_in_image_fast(file_stream, known_face_names, known_face_encodings):
    logging.debug("step1: Load the uploaded image file")
    image = face_recognition.load_image_file(file_stream)

    logging.debug("step2: Resize image to 1/4 size for faster face recognition processing")
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    logging.debug("step3: Find all the faces and face encodings in the current frame of video")
    face_locations = face_recognition.face_locations(small_image, model='hog')

    logging.debug("step4: Get face encodings for any faces in the uploaded image")
    face_encodings = face_recognition.face_encodings(small_image, face_locations)


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

