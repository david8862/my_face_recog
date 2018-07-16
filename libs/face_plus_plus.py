#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2018/7/6 8:49 PM
# @Author : jiewa2
# -*- coding: utf-8 -*-

import urllib2
import json
import time

##################################################

http_url='https://api-cn.faceplusplus.com/facepp/v3/detect'
key = "7WPUZt1wZur09UCfpwRyUpbRWhiHj0pp"
secret = "tpyhJuI456PV9U21wHUjKI_k9yxU8C0b"
return_attributes = 'gender,age,emotion,beauty,skinstatus'
boundary = '----------%s' % hex(int(time.time() * 1000))
data = []
###################################################

def append_header():
    global data
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)

    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)

    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    data.append(return_attributes)

def append_image(filepath):
    global  data
    data.append('--%s' % boundary)
    fr=open(filepath,'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s--\r\n' % boundary)

def send_request():
    #发送POST请求
    http_body='\r\n'.join(data)
    req=urllib2.Request(http_url)
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
    req.add_data(http_body)
    qrcont = None
    try:
        resp = urllib2.urlopen(req, timeout=15)
        qrcont=resp.read()

    except urllib2.HTTPError as e:
        print e.read()
    json_resp = json.loads(qrcont)
    num = len(json_resp['faces']) #人脸个数
    face_s = json_resp['faces']
    face_datas = []
    for i in range(num):
        tempface = face_s[i]
        data2=tempface['attributes']
        face_data = {
            'gender':data2['gender'].values()[0],
            'age':data2['age'].values()[0],
            'location':tempface['face_rectangle']['left']
        }

        emotion = data2['emotion']
        face_data['emotion'] =max(emotion, key=emotion.get)

        score = data2['beauty']
        if face_data['gender']  == 'Male':
            face_data['score'] = score['male_score']
        else:
            face_data['score'] = score['female_score']

        skin = data2['skinstatus']
        face_data['skin'] = max(skin, key=skin.get)

        face_datas.append(face_data)

    face_datas.sort(key=lambda k: k['location'])

    return face_datas


def get_external_result(file):
    global data
    data = []
    append_header()
    append_image(file)
    return send_request()

if __name__ == "__main__":
    face_datas = get_external_result('../imageToSave.png')
    #result_parser(face_datas)
