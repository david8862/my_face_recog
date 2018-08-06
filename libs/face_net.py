#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

from scipy import misc
import PIL.Image
import tensorflow as tf
import numpy as np
import os, copy, logging
from queue import Queue
from threading import Thread
import dlib
import openface
import facenet.src.facenet as facenet
import facenet.src.align.detect_face as detect_face
from .utils import image_files_in_folder, LOG_FORMAT

MODEL=os.path.split(os.path.realpath(__file__))[0] + '/models/20180402-114759.pb'
PREDICTOR_MODEL=os.path.split(os.path.realpath(__file__))[0] + '/models/shape_predictor_68_face_landmarks.dat'
FACE_SIZE=160
MARGIN=44
DISTANCE_THRESHOLD=1.0

face_aligner = None
#LANDMARKIND=openface.AlignDlib.OUTER_EYES_AND_NOSE
LANDMARKIND=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

pnet = None
rnet = None
onet = None
encoding_in_queue = None
encoding_out_queue = None

def init_model():
    init_face_detection_network()
    init_face_aligner()
    init_face_encoding_thread()

def init_face_detection_network():
    global pnet, rnet, onet
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

def init_face_aligner():
    global face_aligner
    face_aligner = openface.AlignDlib(PREDICTOR_MODEL)

def init_face_encoding_thread(model=MODEL):
    global encoding_in_queue,encoding_out_queue
    encoding_in_queue = Queue()
    encoding_out_queue = Queue()
    enc_thread = Thread(target=face_encoding_thread, args=(encoding_in_queue,encoding_out_queue,model))
    enc_thread.start()

# Thread to handle face encoding request
def face_encoding_thread(in_queue, out_queue, model):
    face_encodings = []
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the model
            facenet.load_model(model)
        while True:
            images = in_queue.get()
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            face_encodings = sess.run(embeddings, feed_dict=feed_dict)
            out_queue.put(face_encodings)


def get_face_encodings(images, model=MODEL):
    global encoding_in_queue,encoding_out_queue
    face_encodings = []
    if len(images) == 0:
        return face_encodings

    encoding_in_queue.put(images)
    face_encodings = encoding_out_queue.get()
    return face_encodings


def load_image_file(file_stream, mode='RGB'):
    im = PIL.Image.open(file_stream)
    if mode:
        im = im.convert(mode)
    return np.array(im)

def get_face_locations(image, model=None):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    bounding_boxes = bounding_boxes[:,0:4]
    return [[int(top), int(right), int(bottom), int(left)] for (left, top, right, bottom) in bounding_boxes]


#def get_aligned_face(image, face_location, face_size=FACE_SIZE, margin=MARGIN):
    #global face_aligner
    ## face_location : [top, right, bottom, left]
    #face_rect = dlib.rectangle(left=face_location[3], top=face_location[0], right=face_location[1], bottom=face_location[2])
    ## Use openface to calculate and perform the face alignment
    #alignedFace = face_aligner.align(FACE_SIZE, image, face_rect, landmarkIndices=LANDMARKIND)
    #prewhitened = facenet.prewhiten(alignedFace)
    #return prewhitened


def get_aligned_face(image, face_location, face_size=FACE_SIZE, margin=MARGIN):
    image_size = np.asarray(image.shape)[0:2]
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(face_location[0]-margin/2, 0)
    bb[1] = np.minimum(face_location[1]+margin/2, image_size[1])
    bb[2] = np.minimum(face_location[2]+margin/2, image_size[0])
    bb[3] = np.maximum(face_location[3]-margin/2, 0)
    cropped = image[bb[0]:bb[2],bb[3]:bb[1],:]
    resized = misc.imresize(cropped, (face_size, face_size), interp='bilinear')
    prewhitened = facenet.prewhiten(resized)
    return prewhitened


def get_face_images(image, face_locations):
    img_list = []
    images = []
    for face_location in face_locations:
        aligned_face = get_aligned_face(image, face_location)
        img_list.append(aligned_face)

    if len(img_list) > 0:
        images = np.stack(img_list)
    return images


def get_face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def load_and_align_db(image_paths):
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    images = []

    for image_file in tmp_image_paths:
        image = load_image_file(image_file)
        face_locations = get_face_locations(image)
        if len(face_locations) < 1:
            print("WARNING: No faces found in {}. Ignoring file.".format(image_file))
            image_paths.remove(image_file)
            continue
        elif len(face_locations) > 1:
            print("WARNING: More than one face found in {}. Only considering the first face.".format(file))
        aligned_face = get_aligned_face(image, face_locations[0])
        img_list.append(aligned_face)
    if len(img_list) > 0:
        images = np.stack(img_list)
    return images


def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    image_files = image_files_in_folder(known_people_folder)
    aligned_images = load_and_align_db(image_files)
    for file in image_files:
        basename = os.path.splitext(os.path.basename(file))[0]
        known_names.append(basename)
    known_face_encodings = get_face_encodings(aligned_images)
    return known_names, list(known_face_encodings)


def recognize_faces_in_image(file_stream, known_face_names, known_face_encodings):
    logging.debug("step1: Load the uploaded image file")
    image = load_image_file(file_stream)
    logging.debug("step2: Find all the faces ")
    face_locations = get_face_locations(image)
    face_images = get_face_images(image, face_locations)
    logging.debug("step3: Get face encodings ")
    face_encodings = get_face_encodings(face_images)
    logging.debug("step4: Get face encodings...done")

    face_names = []
    face_distances = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        distances = get_face_distance(known_face_encodings, face_encoding)
        result = list(distances <= DISTANCE_THRESHOLD)
        index = np.argmin(distances)
        name = "  "
        distance = min(distances)
        # If a match was found in known_face_encodings, just use the first one.
        if True in result:
            name = known_face_names[index]

        face_names.append(name)
        face_distances.append(distance)

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

