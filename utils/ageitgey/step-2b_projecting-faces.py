import sys
import dlib
import openface
import argparse
from skimage import io

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "/Users/xiaobizh/.local/virtualenvs/ml/lib/python2.7/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat"

#LANDMARKIND=openface.AlignDlib.OUTER_EYES_AND_NOSE
LANDMARKIND=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP

def main(args):
    # Take the image file name from the command line
    file_name = args.image_file

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)

    win = dlib.image_window()

    # Take the image file name from the command line
    file_name = sys.argv[1]

    # Load the image
    #image = cv2.imread(file_name)
    image = io.imread(file_name)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):

        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)

        # Use openface to calculate and perform the face alignment
        alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=LANDMARKIND)

        # Save the aligned image to a file
        #cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)
        io.imsave("aligned_face_{}.jpg".format(i), alignedFace)

        # Show the desktop window with the image
        win.set_image(alignedFace)
        dlib.hit_enter_to_continue()



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file', type=str, help='Images to check')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
