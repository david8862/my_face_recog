#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 2018/7/6 8:49 PM
# @Author : jiewa2
# -*- coding: utf-8 -*-

import requests
import json
import base64

http_url='https://api-cn.faceplusplus.com/facepp/v3/detect'
key = "7WPUZt1wZur09UCfpwRyUpbRWhiHj0pp"
secret = "tpyhJuI456PV9U21wHUjKI_k9yxU8C0b"
return_attributes = 'gender,age,emotion,beauty,skinstatus'

def get_external_result(encoded_string):

    data = {
        'api_key': key,
        'api_secret': secret,
        'return_attributes': return_attributes,
        'image_base64': encoded_string
    }
    r = requests.post(http_url, data)
    json_resp = r.json()
    num = len(json_resp['faces'])  # 人脸个数
    face_s = json_resp['faces']
    face_datas = []
    for i in range(num):
        tempface = face_s[i]
        data2 = tempface['attributes']
        face_data = {
            'gender':data2['gender']['value'],
            'age':data2['age']['value'],
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



if __name__ == "__main__":
    with open('../imageToSave.png', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        face_datas = get_external_result(encoded_string)
        print(json.dumps(face_datas, sort_keys=True, indent=4, separators=(',', ': ')))
