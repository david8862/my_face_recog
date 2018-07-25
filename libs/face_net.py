#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performs face alignment and calculates L2 distance between the embeddings of images."""

#from __future__ import absolute_import
#from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet.src.facenet as facenet
import facenet.src.align.detect_face as detect_face
from utils import image_files_in_folder

MODEL='models/20180402-114759.pb'


def scan_known_people(known_people_folder, model=MODEL):
    known_names = []
    known_face_encodings = []

    image_files = image_files_in_folder(known_people_folder)
    images = load_and_align_data(image_files, image_size=160, margin=44, gpu_memory_fraction=1.0)

    for file in image_files:
        basename = os.path.splitext(os.path.basename(file))[0]
        known_names.append(basename)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            known_face_encodings = sess.run(embeddings, feed_dict=feed_dict)
            
            nrof_images = len(image_files)
            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, image_files[i]))
            print('')

    print(type(known_face_encodings))
    print(known_face_encodings.shape)
    print(list(known_face_encodings))

    return known_names, list(known_face_encodings)


def recognize_faces_in_image(file_stream, known_face_names, known_face_encodings, model='hog'):
    logging.debug("step1: Load the uploaded image file")
    image = face_recognition.load_image_file(file_stream)
    logging.debug("step2: Find all the faces ")
    face_locations = face_recognition.face_locations(image, model=model)
    logging.debug("step3: Get face encodings ")
    face_encodings = face_recognition.face_encodings(image, face_locations)
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
        result = list(distances <= 0.45)
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
    PRE_FACE_NAMES = face_names
    PRE_LOCATIONS = face_locations
    PRE_DISTANCE = face_distances

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

            
            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        elif len(bounding_boxes) > 1:
            print("got more than 1 face in ", image)
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


#def main(args):

    #images, image_files = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    #with tf.Graph().as_default():

        #with tf.Session() as sess:
      
            ## Load the model
            #facenet.load_model(args.model)
    
            ## Get input and output tensors
            #images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            #embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            #phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            ## Run forward pass to calculate embeddings
            #feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            #emb = sess.run(embeddings, feed_dict=feed_dict)
            
            #nrof_images = len(args.image_files)

            #print('Images:')
            #for i in range(nrof_images):
                #print('%1d: %s' % (i, args.image_files[i]))
            #print('')

            #print(type(emb))
            #print(emb.shape)
            #print(emb)
            
            ## Print distance matrix
            #print('Distance matrix')
            #print('    ', end='')
            #for i in range(nrof_images):
                #print('    %1d     ' % i, end='')
            #print('')
            #for i in range(nrof_images):
                #print('%1d  ' % i, end='')
                #for j in range(nrof_images):
                    ##dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    #dist = np.linalg.norm(emb[i,:] - emb[j,:])
                    #print('  %1.4f  ' % dist, end='')
                #print('')


#def parse_arguments(argv):
    #parser = argparse.ArgumentParser()
    
    #parser.add_argument('model', type=str, 
        #help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    #parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    #parser.add_argument('--image_size', type=int,
        #help='Image size (height, width) in pixels.', default=160)
    #parser.add_argument('--margin', type=int,
        #help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    #parser.add_argument('--gpu_memory_fraction', type=float,
        #help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    #return parser.parse_args(argv)

#if __name__ == '__main__':
    #main(parse_arguments(sys.argv[1:]))
