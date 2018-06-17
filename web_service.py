# This is a simple web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains faces
# of CRDC TIPBU member and the face location in image.
# The result is returned as json. For example:
#
# $ curl -XPOST -F "file=@test.jpg" http://0.0.0.0:5001/phoneapi
#
# Returns:
#
#{
#    "face_found_in_image": true,
#    "face_data": {
#        "face1": {
#            "name": "conli", 
#            "bottom": 494, 
#            "left": 93, 
#            "right": 129, 
#            "top": 458
#        }, 
#    }, 
#}
#
# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/


import numpy as np
import cv2, PIL
from flask import Flask, jsonify, request, redirect, make_response, render_template, Response
from libs.faces import recognize_faces_in_image, recognize_faces_in_image_fast, allowed_image, scan_known_people


app = Flask(__name__)

origin_image_buffer = np.zeros(5)
result_image_buffer = np.zeros(5)


@app.route('/phoneapi', methods=['GET', 'POST'])
def handle_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_image(file.filename):
            # The image file seems valid! Detect faces and return the result.
            result = recognize_faces_in_image(file, known_face_names, known_face_encodings)
            return jsonify(result)

    # If no valid image file was uploaded, show the file upload form:
    return jsonify({
        "face_found_in_image": False,
        "face_data": {},
    })


def add_face_rectangle(result, imgfile):
    image = PIL.Image.open(imgfile)
    img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

    if result["face_found_in_image"] == False:
        return img

    #for face in result["face_data"]:
    for (index, face) in  result["face_data"].items():
        left = face["left"]
        right = face["right"]
        top = face["top"]
        bottom = face["bottom"]
        name = face["name"]

        # Draw a box around the face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return img

def save_ori_image(imgfile):
    image = PIL.Image.open(imgfile)
    img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    return img


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    global result_image_buffer, origin_image_buffer
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_image(file.filename):
            # The image file seems valid! Detect faces and return the result.
            result = recognize_faces_in_image_fast(file, known_face_names, known_face_encodings)
            img = add_face_rectangle(result, file)
            retval, buff = cv2.imencode('.jpg', img)
            result_image_buffer = buff.copy()

            ori_img = save_ori_image(file)
            retval, ori_buff = cv2.imencode('.jpg', ori_img)
            origin_image_buffer = ori_buff.copy()
            return render_template('showimage.html')

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Is this a people of CRDC TIPBU?</title>
    <h1>Upload a picture and see if you're in CRDC TIPBU family!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

@app.route('/result_image')
def get_result_image():
    global result_image_buffer
    frame = result_image_buffer.tobytes()
    return Response(frame, mimetype='image/jpg')

@app.route('/origin_image')
def get_origin_image():
    global origin_image_buffer
    frame = origin_image_buffer.tobytes()
    return Response(frame, mimetype='image/jpg')


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == "__main__":
    known_face_names, known_face_encodings = scan_known_people("face_db")
    app.run(host='0.0.0.0', port=5001, debug=True)
