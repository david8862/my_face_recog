#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, base64, cv2
#import keras
#from keras.models import load_model
import tensorflow as tf
import numpy as np
import face_recognition
from queue import Queue
from threading import Thread


class Face_Features(object):
    emotion_model = os.path.split(os.path.realpath(__file__))[0] + '/models/fer2013_mini_XCEPTION_Dense.79-0.66.hdf5'
    gender_model = os.path.split(os.path.realpath(__file__))[0] + '/models/gender_mini_XCEPTION.21-0.95.hdf5'
    gender_offsets = (30, 60)
    emotion_offsets = (20, 40)

    def __init__(self):
        self.__init_face_features_thread()

    def __init_face_features_thread(self):
        self.features_in_queue = Queue()
        self.features_out_queue = Queue()
        self.features_thread = Thread(target=self.__face_features_thread, args=(self.features_in_queue, self.features_out_queue))
        self.features_thread.start()

    # Thread to handle face features request
    def __face_features_thread(self, in_queue, out_queue):
        # loading models
        tf.keras.backend.clear_session()
        self.emotion_classifier = tf.keras.models.load_model(self.emotion_model, compile=False)
        self.gender_classifier = tf.keras.models.load_model(self.gender_model, compile=False)

        # loading labels
        self.emotion_labels = self.__get_labels('fer2013')
        self.gender_labels = self.__get_labels('imdb')

        # getting input model shapes for inference
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.gender_target_size = self.gender_classifier.input_shape[1:3]

        while True:
            encoded_string = in_queue.get()
            face_datas = self.__generate_features(encoded_string)
            out_queue.put(face_datas)

    def get_features(self, encoded_string):
        self.features_in_queue.put(encoded_string)
        face_datas = self.features_out_queue.get()
        return face_datas

    def __get_labels(self, dataset_name):
        if dataset_name == 'fer2013':
            return {0:'angry',1:'disgust',2:'fear',3:'happy',
                    4:'sad',5:'surprise',6:'neutral'}
        elif dataset_name == 'imdb':
            return {0:'woman', 1:'man'}
        else:
            raise Exception('Invalid dataset name')

    def __preprocess_input(self, x, v2=True):
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    def __apply_offsets(self, face_coordinates, offsets):
        x, y, width, height = face_coordinates
        x_off, y_off = offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


    def __generate_features(self, encoded_string):
        img = base64.b64decode(encoded_string)
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        #gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rgb_image = img

        faces = face_recognition.face_locations(gray_image, model='hog')
        faces.sort(key=lambda k:k[3])

        face_datas = []

        for face_coordinates in faces:
            top, right, bottom, left = face_coordinates
            face_coordinates = (left, top, right - left, bottom - top)

            x1, x2, y1, y2 = self.__apply_offsets(face_coordinates, self.gender_offsets)
            rgb_face = rgb_image[y1:y2, x1:x2]

            x1, x2, y1, y2 = self.__apply_offsets(face_coordinates, self.emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                rgb_face = cv2.resize(rgb_face, (self.gender_target_size))
                gray_face = cv2.resize(gray_face, (self.emotion_target_size))
            except:
                continue
            gray_face = self.__preprocess_input(gray_face, False)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(self.emotion_classifier.predict(gray_face))
            emotion_text = self.emotion_labels[emotion_label_arg]

            rgb_face = np.expand_dims(rgb_face, 0)
            rgb_face = self.__preprocess_input(rgb_face, False)
            #gender_prediction = gender_classifier.predict(gray_face)
            gender_label_arg = np.argmax(self.gender_classifier.predict(gray_face))
            gender_text = self.gender_labels[gender_label_arg]

            face_data = {
                'gender':gender_text,
                'emotion':emotion_text,
                'score': -1,
                'age': -1,
                'skin': 'NA',
            }
            face_datas.append(face_data)
        return face_datas


if __name__ == "__main__":
    with open(os.path.split(os.path.realpath(__file__))[0] + '/../face_test/test1.jpg', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        face_feature = Face_Features()
        face_datas = face_feature.get_features(encoded_string)
        print(json.dumps(face_datas, sort_keys=True, indent=4, separators=(',', ': ')))
