#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

from scipy import misc
import PIL.Image
import tensorflow as tf
import numpy as np
import sys, os, re, copy, logging
from queue import Queue
from threading import Thread
import dlib
import openface
import facenet.src.facenet as facenet
import facenet.src.align.detect_face as detect_face
#from .utils import image_files_in_folder, LOG_FORMAT


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png|gif|bmp)', f, flags=re.I)]

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class Face(object):
    def __init__(self):
        self.name = None
        self.container_path = None
        self.container_image = None
        self.bounding_box = None
        self.image = None
        self.embedding = None


class Face_Recognition(object):
    distance_threshold=1.0

    def __init__(self):
        self.detect = Detection()
        self.align = Aligner()
        self.encoder = Encoder()
        self.facedb = []

    def load_image_file(self, file_stream, mode='RGB'):
        im = PIL.Image.open(file_stream)
        if mode:
            im = im.convert(mode)
        return np.array(im)

    def scan_known_people(self, known_people_folder):
        image_files = image_files_in_folder(known_people_folder)
        for file in image_files:
            person_name = os.path.splitext(os.path.basename(file))[0]
            image = self.load_image_file(file)
            locations = self.detect.get_face_locations(image)

            if len(locations) < 1:
                print("WARNING: No faces found in {}. Ignoring file.".format(file))
                #image_paths.remove(image_file)
                continue
            elif len(locations) > 1:
                print("WARNING: More than one face found in {}. Only considering the first face.".format(file))

            aligned_face = self.align.get_aligned_face(image, locations[0])
            face = Face()
            face.name = person_name
            face.bounding_box = locations[0]
            face.image = aligned_face
            face.container_image = image
            face.container_path = file
            face.embedding = self.encoder.generate_embedding(aligned_face)
            self.facedb.append(face)

    def get_face_distance(self, face_embeddings, face_to_compare):
        if len(face_embeddings) == 0:
            return np.empty((0))
        return np.linalg.norm(face_embeddings - face_to_compare, axis=1)


    def recognize_faces_in_image(self, image_stream):
        logging.debug("step1: Load the image")
        if isinstance (image_stream, np.ndarray):
            image = image_stream
        else:
            image = self.load_image_file(image_stream)
        logging.debug("step2: Find all the faces ")
        face_locations = self.detect.get_face_locations(image)
        face_images = self.align.get_aligned_faces(image, face_locations)
        logging.debug("step3: Get face encodings ")
        face_embeddings = self.encoder.generate_embeddings(face_images)
        logging.debug("step4: Get face encodings...done")

        face_names = []
        face_distances = []
        facedb_embeddings = np.reshape([face.embedding for face in self.facedb], (len(self.facedb), -1))
        facedb_names = [face.name for face in self.facedb]
        for face_embedding in face_embeddings:
            # See if the face is a match for the known face(s)
            distances = self.get_face_distance(facedb_embeddings, face_embedding)
            result = list(distances <= self.distance_threshold)
            index = np.argmin(distances)
            name = "  "
            distance = min(distances)
            # If a match was found in known_face_encodings, just use the first one.
            if True in result:
                name = facedb_names[index]
            face_names.append(name)
            face_distances.append(distance)

        face_found = False
        if len(face_embeddings) > 0:
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

class Detection(object):
    # face detection parameters
    gpu_memory_fraction = 0.3
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self):
        self.pnet, self.rnet, self.onet = self.__setup_mtcnn()

    def load_image_file(self, file_stream, mode='RGB'):
        im = PIL.Image.open(file_stream)
        if mode:
            im = im.convert(mode)
        return np.array(im)

    def __setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, None)

    def get_face_locations(self, image, model=None):
        bounding_boxes, _ = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        bounding_boxes = bounding_boxes[:,0:4]
        return [[int(top), int(right), int(bottom), int(left)] for (left, top, right, bottom) in bounding_boxes]


class Aligner(object):
    face_size=160
    margin=44
    predictor_model=os.path.split(os.path.realpath(__file__))[0] + '/models/shape_predictor_68_face_landmarks.dat'
    landmark=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    #landmark=openface.AlignDlib.OUTER_EYES_AND_NOSE

    def __init__(self):
        self.face_aligner = openface.AlignDlib(self.predictor_model)

    def load_image_file(self, file_stream, mode='RGB'):
        im = PIL.Image.open(file_stream)
        if mode:
            im = im.convert(mode)
        return np.array(im)

    def get_aligned_face(self, image, face_location, face_size=None, margin=None, model='shape_predictor'):
        if face_size is None:
            face_size = self.face_size
        if margin is None:
            margin = self.margin

        if model == 'shape_predictor':
            # face_location : [top, right, bottom, left]
            face_rect = dlib.rectangle(left=face_location[3], top=face_location[0], right=face_location[1], bottom=face_location[2])
            # Use openface to calculate and perform the face alignment
            alignedFace = self.face_aligner.align(face_size, image, face_rect, landmarkIndices=self.landmark)
            return alignedFace
        elif model == 'resize':
            image_size = np.asarray(image.shape)[0:2]
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(face_location[0]-margin/2, 0)
            bb[1] = np.minimum(face_location[1]+margin/2, image_size[1])
            bb[2] = np.minimum(face_location[2]+margin/2, image_size[0])
            bb[3] = np.maximum(face_location[3]-margin/2, 0)
            cropped = image[bb[0]:bb[2],bb[3]:bb[1],:]
            resized = misc.imresize(cropped, (face_size, face_size), interp='bilinear')
            return resized
        else:
            raise ValueError("Invalid align model type. Supported models are ['shape_predictor', 'resize'].")

    def get_aligned_faces(self, image, face_locations, model='shape_predictor'):
        img_list = []
        images = []
        for face_location in face_locations:
            aligned_face = self.get_aligned_face(image, face_location, model=model)
            img_list.append(aligned_face)

        if len(img_list) > 0:
            images = np.stack(img_list)
        return images



class Encoder(object):
    model=os.path.split(os.path.realpath(__file__))[0] + '/models/20180402-114759.pb'

    def __init__(self, model=None):
        if model is None:
            model = self.model
        self.__init_face_embedding_thread(model)

    def __init_face_embedding_thread(self, model=None):
        if model is None:
            model = self.model
        self.embedding_in_queue = Queue()
        self.embedding_out_queue = Queue()
        self.embedding_thread = Thread(target=self.__face_embedding_thread, args=(self.embedding_in_queue, self.embedding_out_queue, model))
        self.embedding_thread.start()

    # Thread to handle face encoding request
    def __face_embedding_thread(self, in_queue, out_queue, model):
        face_embeddings = []
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                # Load the model
                facenet.load_model(model)
            while True:
                face_image = in_queue.get()
                prewhiten_face = facenet.prewhiten(face_image)
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                # Run forward pass to calculate embeddings
                feed_dict = { images_placeholder: [prewhiten_face], phase_train_placeholder:False }
                face_embeddings = sess.run(embeddings, feed_dict=feed_dict)
                out_queue.put(face_embeddings)

    def generate_embedding(self, face_image):
        embedding = []
        if len(face_image) == 0:
            return embedding

        self.embedding_in_queue.put(face_image)
        embedding = self.embedding_out_queue.get()
        return embedding

    def generate_embeddings(self, face_images):
        embedding_list = []
        for face_image in face_images:
            embedding = self.generate_embedding(face_image)
            embedding_list.append(embedding)
        if len(embedding_list) > 0:
            embeddings = np.stack(embedding_list)
        return embedding_list


import argparse
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
            result = recognition.recognize_faces_in_image(small_frame)
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
