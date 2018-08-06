import sys
import dlib
import argparse
import numpy as np
from skimage import io


# You can download the required pre-trained face detection & recognition model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
# http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
predictor_68_point_model = "/Users/xiaobizh/.local/virtualenvs/py3ml/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat"
predictor_5_point_model = "/Users/xiaobizh/.local/virtualenvs/py3ml/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_5_face_landmarks.dat"
face_recognition_model = "/Users/xiaobizh/.local/virtualenvs/py3ml/lib/python3.6/site-packages/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat"

MODEL='large'


def main(args):
    # Take the image file name from the command line
    file_name = args.image_file

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    if MODEL == 'large':
        face_pose_predictor = dlib.shape_predictor(predictor_68_point_model)
    elif MODEL == 'small':
        face_pose_predictor = dlib.shape_predictor(predictor_5_point_model)
    else:
        raise ValueError("Invalid detector model type. Supported models are ['large', 'small'].")
    face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

    win = dlib.image_window()

    # Take the image file name from the command line
    file_name = sys.argv[1]

    # Load the image
    image = io.imread(file_name)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

    # Show the desktop window with the image
    win.set_image(image)

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):

        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("* Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

        # Draw a box around each face we found
        win.add_overlay(face_rect)


        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)

        # Draw the face landmarks on the screen.
        win.add_overlay(pose_landmarks)

        # Get the the face's encoding and convert to numpy array
        face_encoding = face_encoder.compute_face_descriptor(image, pose_landmarks, num_jitters=1)
        face_encoding = np.array(face_encoding)
        print("* Face #{}'s encoding:\n{}".format(i, face_encoding))

    dlib.hit_enter_to_continue()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file', type=str, help='Images to check')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
