#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This is a secure web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains faces
# of CRDC TIPBU member and the face location in image.
# The result is returned as json. For example:
#
# $ curl -k -XPOST -F "file=@test.jpg" https://0.0.0.0:5001/phoneapi
#
# $ curl -m 1 -k -F "mac=5006AB802B51" -F "cec=xiaobizh" https://localhost:5001/emlogin/demo
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
import cv2, PIL, os
from flask import Flask, jsonify, request, redirect, make_response, render_template, Response
from libs.faces import init_model, scan_known_people, recognize_faces_in_image
#from libs.face_net import init_model, scan_known_people, recognize_faces_in_image
from libs.utils import allowed_image
from libs.face_plus_plus import get_external_result
from libs.cucm import Cucm


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

origin_image_buffer = np.zeros(5)
result_image_buffer = np.zeros(5)
known_face_names = []
known_face_encodings = []


def update_cucm_info(phone_mac, cec_id):
    cucm_host = "10.74.63.21"
    username = "1"
    password = "1"

    data_dict  = {
        "xiaobizh": {"DN": "10710", "name": "Xiaobin Zhang"},
        "hoqiu": {"DN": "10711", "name": "Fans Qiu"},
        "jujin": {"DN": "10712", "name": "Jun Jin"},
        "jiewa2": {"DN": "10713", "name": "Jie Wang"},
    }

    line_index = "Line [1] -"

    my_cucm = Cucm(host=cucm_host, cm_username=username, cm_password=password)
    my_phone = my_cucm.find_phone(phone_mac)
    my_cucm.change_line_dn(my_phone, line_index, data_dict[cec_id]["DN"])
    my_cucm.change_line_label(my_phone, data_dict[cec_id]["DN"], data_dict[cec_id]["name"])
    my_cucm.quit()


@app.route('/emlogin/demo', methods=['GET', 'POST'])
def emlogin_demo():
    if request.method != 'POST':
        return 'No data post!\n'

    if 'mac' not in request.form:
        return 'No MAC addr received!\n'
    if 'cec' not in request.form:
        return 'No CEC ID received!\n'

    cec = request.form['cec']
    mac = request.form['mac']
    update_cucm_info(mac, cec)

    return 'Update successfully!\n'


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
def upload_image_manually():
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
            result = recognize_faces_in_image(file, known_face_names, known_face_encodings)
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



@app.route('/capture', methods=['GET', 'POST'])
def upload_image():
    import base64, uuid
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST' and 'photo' in request.form:
        filename = uuid.uuid1()
        if 'cec_id' in request.form and request.form['cec_id'] != '':
            filename = "{}{}".format(filename,request.form['cec_id'])
        upg_photo = request.form['photo'].split(',')[-1]
        file = os.path.join(APP_ROOT, "face_test/{}.png".format(filename))
        with open(file, "wb") as fh:
            fh.write(base64.b64decode(upg_photo))
        result = recognize_faces_in_image(file, known_face_names, known_face_encodings)
        faces = list(result['face_data'].values())
        faces.sort(key=lambda k:k['left'])
        if 'enable_face_plus' in request.form and request.form['enable_face_plus'] == 'on':
            face_datas = get_external_result(upg_photo)
            for i in range(len(faces)):
                face_datas[i]['name'] = faces[i]['name']
            return render_template('result.html',face_datas = face_datas)

        return render_template('result.html',face_datas = faces)

if __name__ == "__main__":
    init_model()
    face_db = os.path.join(APP_ROOT, "face_db")
    known_face_names, known_face_encodings = scan_known_people(face_db)
    app.run(host='0.0.0.0', port=5001, ssl_context='adhoc')
