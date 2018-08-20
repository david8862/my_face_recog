# This is a demo of running face detect on live video from your webcam, which includes some
# basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

import face_recognition
import cv2

MODEL="large"

def face_detect():
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    # Initialize some variables
    face_locations = []
    face_landmarks_list = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(frame, model='hog')
            #face_locations = face_recognition.face_locations(frame, model='cnn')
            face_landmarks_list = face_recognition.face_landmarks(frame, model=MODEL)
        process_this_frame = not process_this_frame

        # Display the results
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            #top *= 4
            #right *= 4
            #bottom *= 4
            #left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        for face_landmarks in face_landmarks_list:
            # Print the location of each facial feature in this image
            if MODEL == 'large':
                facial_features = [
                    'chin',
                    'left_eyebrow',
                    'right_eyebrow',
                    'nose_bridge',
                    'nose_tip',
                    'left_eye',
                    'right_eye',
                    'top_lip',
                    'bottom_lip'
                ]
            elif MODEL == 'small':
                 facial_features = [
                    'nose_tip',
                    'left_eye',
                    'right_eye',
                ]
            else:
                raise ValueError("Invalid landmarks model type. Supported models are ['small', 'large'].")

            for facial_feature in facial_features:
            #for facial_feature in ['top_lip', 'bottom_lip']:
                for (x,y) in face_landmarks[facial_feature]:
                    #x *= 4
                    #y *= 4
                    cv2.circle(frame, (x,y), 4, (0, 0, 255), -1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    face_detect()


if __name__ == "__main__":
    main()
