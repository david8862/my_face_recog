import sys
import dlib
from skimage import io


cnn_face_detector_model = "/Users/xiaobizh/.local/virtualenvs/ml/lib/python2.7/site-packages/face_recognition_models/models/mmod_human_face_detector.dat"

MODEL='cnn'

# Take the image file name from the command line
file_name = sys.argv[1]

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_model)

win = dlib.image_window()

# Load the image into an array
image = io.imread(file_name)

# Run the face detector on the image data.
# The result will be the bounding boxes of the faces in our image.
if MODEL == 'hog':
    detected_faces = face_detector(image, 1)
elif MODEL == 'cnn':
    detected_faces = cnn_face_detector(image, 1)
    detected_faces = [face.rect for face in detected_faces]
else:
    raise ValueError("Invalid detector model type. Supported models are ['hog', 'cnn'].")

print("I found {} faces in the file {}".format(len(detected_faces), file_name))

# Open a window on the desktop showing the image
win.set_image(image)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):
    # Detected faces are returned as an object with the coordinates 
    # of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

    # Draw a box around each face we found
    win.add_overlay(face_rect)
	        
# Wait until the user hits <enter> to close the window	        
dlib.hit_enter_to_continue()
