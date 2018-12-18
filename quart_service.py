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
import cv2, PIL, os, re
from quart import Quart, jsonify, request, redirect, make_response, render_template, Response
#import flask_monitoringdashboard as dashboard
import requests, json
from libs.faces import Face_Recognition
#from libs.face_net import Face_Recognition
from libs.utils import allowed_image
from libs.face_plus_plus import get_external_result
from libs.cucm import Cucm


app = Quart(__name__)
#dashboard.bind(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

origin_image_buffer = np.zeros(5)
result_image_buffer = np.zeros(5)
recognition = None


def update_cucm_info(phone_mac, cec_id):
    cucm_host = "10.74.63.21"
    username = "1"
    password = "1"

    data_dict = {
        #"xiaobizh": {"DN": "10710", "name": "Xiaobin Zhang"},
        #"hoqiu": {"DN": "10711", "name": "Fans Qiu"},
        #"jujin": {"DN": "10712", "name": "Jun Jin"},
        #"jiewa2": {"DN": "10713", "name": "Jie Wang"},

        "crobbins": {"DN": "10600", "name": "Chuck Robbins"},
        "amychang": {"DN": "10610", "name": "Amy Chang"},
        "tpuorro": {"DN": "10620", "name": "Tom Puorro"},
        "naim": {"DN": "10630", "name": "Hakim Mehmood"},
        "denwu": {"DN": "10710", "name": "Dennis Wu"},
        "choli": {"DN": "10711", "name": "Tracy Li"},
        "asuliu": {"DN": "10712", "name": "Asura Liu"},
        "xiaowang": {"DN": "10713", "name": "Bit Wang"},
        "bolei2": {"DN": "10714", "name": "Bo Lei"},
        "chenzhu2": {"DN": "10715", "name": "Chenghao Zhu"},
        "zisu": {"DN": "10716", "name": "zisu"},
        "junbzhan": {"DN": "10717", "name": "Junbo Zhang"},
        "junhma": {"DN": "10718", "name": "Junhua Ma"},
        "migu": {"DN": "10719", "name": "Mingpo Gu"},
        "nyuan": {"DN": "10720", "name": "Ning Yuan"},
        "nzhang2": {"DN": "10721", "name": "Ning Zhang"},
        "qianx": {"DN": "10722", "name": "Qian Xu"},
        "zhzhe": {"DN": "10723", "name": "Rodger Zhao"},
        "siwzhang": {"DN": "10724", "name": "Siwei Zhang"},
        "xiaolihu": {"DN": "10725", "name": "Xiaolin Huang"},
        "xiaomma": {"DN": "10726", "name": "Xiaomin Ma"},
        "xuaxu": {"DN": "10727", "name": "Xuan Xu"},
        "yifding": {"DN": "10728", "name": "Yifan Ding"},
        "yubmao": {"DN": "10729", "name": "Yubing Mao"},
        "zhixu": {"DN": "10730", "name": "Zhier Xu"},
        "biychen": {"DN": "10731", "name": "Biyun Chen"},
        "chanwei": {"DN": "10732", "name": "Changhao Wei"},
        "cheyang2": {"DN": "10733", "name": "Chen Yang"},
        "conli": {"DN": "10734", "name": "Cong Li"},
        "hoqiu": {"DN": "10735", "name": "Fans Qiu"},
        "feixi": {"DN": "10736", "name": "Fei Xie"},
        "jiangfzh": {"DN": "10737", "name": "Jiangfeng Zhu"},
        "jiewa2": {"DN": "10738", "name": "Jie Wang"},
        "kaiche": {"DN": "10739", "name": "Kai Chen"},
        "rilli": {"DN": "10740", "name": "Rilong Li"},
        "xiaobizh": {"DN": "10741", "name": "Xiaobin Zhang"},
        "hxiaxia": {"DN": "10742", "name": "XiaXia He"},
        "zhiholiu": {"DN": "10743", "name": "Zhihong Liu"},
        "zhijin": {"DN": "10744", "name": "Zhimin Jin"},
        "chanwan": {"DN": "10745", "name": "Chan Wang"},
        "huijiang": {"DN": "10746", "name": "Huitao Jiang"},
        "shenghli": {"DN": "10747", "name": "Shenghui Lin"},
        "xiaqin": {"DN": "10748", "name": "Xiaofeng Qin"},
        "xueliang": {"DN": "10749", "name": "Xuebin Liang"},
        "cliu4": {"DN": "10750", "name": "Christopher Liu"},
        "gtie": {"DN": "10751", "name": "Ge Tie"},
        "haniu": {"DN": "10752", "name": "Hao Niu"},
        "huigjin": {"DN": "10753", "name": "Huiguo Jin"},
        "zhiyao": {"DN": "10754", "name": "Jerry Yao"},
        "jianc2": {"DN": "10755", "name": "Jian Chen"},
        "jiazhe": {"DN": "10756", "name": "Jiazhi He"},
        "jibbao": {"DN": "10757", "name": "Jibin Bao"},
        "jingmxu": {"DN": "10758", "name": "Jingming Xu"},
        "jujin": {"DN": "10759", "name": "Jun Jin"},
        "qzhang2": {"DN": "10760", "name": "Justin Zhang"},
        "mengl": {"DN": "10761", "name": "Maggie Li"},
        "payzhu": {"DN": "10762", "name": "Payne Zhu"},
        "qiaji": {"DN": "10763", "name": "Qianhui Ji"},
        "lzhang3": {"DN": "10764", "name": "Ruby Zhang"},
        "xzhao3": {"DN": "10765", "name": "Xin Zhao"},
        "yanmeng": {"DN": "10766", "name": "Yang Meng"},
        "yxue2": {"DN": "10767", "name": "Yi Xue"},
        "yueyu": {"DN": "10768", "name": "Yue Yu"},
        "yuzho2": {"DN": "10769", "name": "Zhou Yu"},
        "huchen2": {"DN": "10770", "name": "Frank Chen"},
        "guozhang": {"DN": "10771", "name": "Guoming Zhang"},
        "hbian": {"DN": "10772", "name": "Huichao Bian"},
        "junhuang": {"DN": "10773", "name": "Juncheng Huang"},
        "liangxwa": {"DN": "10774", "name": "Liangxing Wang"},
        "lingjin": {"DN": "10775", "name": "Lingjiang Jin"},
        "tiqi": {"DN": "10776", "name": "Ting Qi"},
        "ycheng3": {"DN": "10777", "name": "Yan Cheng"},
        "zhpeng": {"DN": "10778", "name": "Zhigang Peng"},
        "haxia": {"DN": "10779", "name": "Harry Xia"},
        "gachen2": {"DN": "10780", "name": "Jerry Chen"},
        "pezhang2": {"DN": "10781", "name": "Peng Zhang"},
        "ronling": {"DN": "10782", "name": "Rongcai Ling"},
        "shuyzhan": {"DN": "10783", "name": "Shuyi Zhang"},
        "xigu": {"DN": "10784", "name": "Xingcai Gu"},
        "yawan": {"DN": "10785", "name": "Yafei Wan"},
        "allren": {"DN": "10786", "name": "Allan Ren"},
        "bruzhang": {"DN": "10787", "name": "Bruce Zhang"},
        "gexue": {"DN": "10788", "name": "George Xue"},
        "fugd": {"DN": "10789", "name": "Guangda Fu"},
        "jameshe": {"DN": "10790", "name": "James He"},
        "jiajiang": {"DN": "10791", "name": "Java Jiang"},
        "jianzo": {"DN": "10792", "name": "Jian Zou"},
        "jiaqshi": {"DN": "10793", "name": "JiaQi Shi"},
        "jingyhua": {"DN": "10794", "name": "Jingyi Huang"},
        "zhoyang": {"DN": "10795", "name": "Joe Yang"},
        "xugwu": {"DN": "10796", "name": "Joly Wu"},
        "pengzho": {"DN": "10797", "name": "Peng Zhou"},
        "shafu": {"DN": "10798", "name": "Rony Fu"},
        "stevyu": {"DN": "10799", "name": "Steve Yu"},
        "xzhuang": {"DN": "10800", "name": "Xue Zhuang"},
        "zhaoli": {"DN": "10801", "name": "Zhaozhuo Li"},
        "riren": {"DN": "10802", "name": "Richard Ren"},
        "jihuo": {"DN": "10803", "name": "Suki Huo"},
        "wenloli": {"DN": "10804", "name": "Wenlong Li"},
        "fanwang2": {"DN": "10805", "name": "Fang Wang"},
        "hongdiz": {"DN": "10806", "name": "Hongdi Zhang"},
        "huanyliu": {"DN": "10807", "name": "Huanyi Liu"},
        "xuameng": {"DN": "10808", "name": "Sandy Meng"},
        "peihchen": {"DN": "10809", "name": "William Peihua Chen"},
        "shjun": {"DN": "10811", "name": "Jun Shu"},
        "wakai": {"DN": "10812", "name": "Kai Wang"},
        "shijzhan": {"DN": "10813", "name": "Shijie Zhang"},
        "shugwang": {"DN": "10814", "name": "Shuguang Wang"},
        "taichifu": {"DN": "10815", "name": "Taichi Fu"},
        "chenyu2": {"DN": "10816", "name": "Chen Yu"},
        "cumu": {"DN": "10817", "name": "cumu"},
        "dandtang": {"DN": "10818", "name": "Dandan Tang"},
        "delzhang": {"DN": "10819", "name": "Deli Zhang"},
        "dongh": {"DN": "10820", "name": "Dong Han"},
        "fangre": {"DN": "10821", "name": "Fang Ren"},
        "haisyang": {"DN": "10822", "name": "Haisheng Yang"},
        "haizou": {"DN": "10823", "name": "Haiyan Zou"},
        "hhuiguan": {"DN": "10824", "name": "Huiguang Huang"},
        "jixu3": {"DN": "10825", "name": "Ji Xu"},
        "jingalin": {"DN": "10826", "name": "Jingan Lin"},
        "jiyou": {"DN": "10827", "name": "Jinlei You"},
        "jisi": {"DN": "10828", "name": "Jinyuan Si"},
        "leiz3": {"DN": "10829", "name": "Lei Zhang"},
        "qiujliu": {"DN": "10830", "name": "Liu Qiuju"},
        "milv": {"DN": "10831", "name": "Lv Mingke"},
        "meig": {"DN": "10832", "name": "Mei Gu"},
        "pengfche": {"DN": "10833", "name": "Pengfei Chen"},
        "pzhai": {"DN": "10834", "name": "Pengwei Zhai"},
        "piwang2": {"DN": "10835", "name": "Ping Wang"},
        "shaliu2": {"DN": "10836", "name": "Sha Liu"},
        "shuoy": {"DN": "10837", "name": "Shuo Yang"},
        "tiren": {"DN": "10838", "name": "Tianlan Ren"},
        "wenca": {"DN": "10839", "name": "Wen Cao"},
        "wenlli": {"DN": "10840", "name": "Wenling Li"},
        "xiaopzh2": {"DN": "10841", "name": "Xiaopeng Zhao"},
        "xinlxu": {"DN": "10842", "name": "Xinli Xu"},
        "xixdong": {"DN": "10843", "name": "Xixi Dong"},
        "yany2": {"DN": "10845", "name": "Yan Yang"},
        "yanahuan": {"DN": "10846", "name": "Yanan Huang"},
        "yangw3": {"DN": "10847", "name": "Yang Wang"},
        "yanlji": {"DN": "10848", "name": "Yanli Ji"},
        "yisun2": {"DN": "10849", "name": "Yi Sun"},
        "yipzou": {"DN": "10850", "name": "Yiping Zou"},
        "zhihzhen": {"DN": "10851", "name": "Zhihui Zheng"},
        "zhiqizha": {"DN": "10852", "name": "Zhiqiang Zhao"},
        "wenjunl": {"DN": "10853", "name": "Wenjun Li"},
        "yingpang": {"DN": "10854", "name": "Ying Pang"},
        "yuwu2": {"DN": "10855", "name": "Yu Wu"},
    }

    line_index = "Line [1] -"

    my_cucm = Cucm(host=cucm_host, cm_username=username, cm_password=password)
    my_phone = my_cucm.find_phone(phone_mac)
    my_cucm.change_line_dn(my_phone, line_index, data_dict[cec_id]["DN"])
    my_cucm.change_line_label(my_phone, data_dict[cec_id]["DN"], data_dict[cec_id]["name"])
    my_cucm.quit()


def checkip(ip):
    p = re.compile('^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$')
    if p.match(ip):
        return True
    else:
        return False


def do_em_login(phone_mac, cec_id, cucm, domain):
    mobile_verify = True
    em_user = 'pd2'
    em_passwd = '88898889'
    #form login url, format like:
    #http://shark-ucm-171.cisco.com:8080/emapp/EMAppServlet?device=SEP0057D2C00CE2&EMCC=true&seq=12345&userid=em1
    #https://shark-ucm-171.cisco.com:8443/emapp/EMAppServlet?device=SEP0057D2C00CE2&EMCC=true&seq=12345&userid=em1

    if checkip(cucm) == True:
        server = cucm
    else:
        server = cucm + '.' + domain

    #url = 'http://10.74.63.21:8080/emapp/EMAppServlet?device=SEP5006AB802B51&EMCC=true&seq=12345&userid=em1'
    url = 'http://' + server + ':8080/emapp/EMAppServlet?device=SEP' + phone_mac + '&EMCC=true&seq=' + em_passwd + '&userid=' + em_user
    #url = 'https://' + server + ':8443/emapp/EMAppServlet?device=SEP' + phone_mac + '&EMCC=true&seq=' + em_passwd + '&userid=' + em_user

    if mobile_verify == True:
        try:
            response = requests.get(url, verify=False)
            if response.status_code != 200:
                print ("em request to ", cucm, " failed!")
                print (json.dumps(response.json(), indent=4, sort_keys=True))
                return False
        except Exception as e:
            print ("em request exception", e)
            return False


@app.route('/emlogin/demo', methods=['GET', 'POST'])
async def emlogin_demo():
    if request.method != 'POST':
        return 'No data post!\n'

    if 'mac' not in request.form:
        return 'No MAC addr received!\n'
    if 'cec' not in request.form:
        return 'No CEC ID received!\n'
    if 'activecucm' not in request.form:
        return 'No CUCM address received!\n'
    if 'domain' not in request.form:
        return 'No domain name received!\n'

    cec = request.form['cec']
    mac = request.form['mac']
    cucm = request.form['activecucm']
    domain = request.form['domain']
    #update_cucm_info(mac, cec)
    do_em_login(mac, cec, cucm, domain)

    return 'Update successfully!\n'


@app.route('/phoneapi', methods=['GET', 'POST'])
async def handle_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        filelist = await request.files
        if 'file' not in filelist:
            return redirect(request.url)

        file = filelist['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_image(file.filename):
            # The image file seems valid! Detect faces and return the result.
            result = recognition.recognize_faces_in_image(file)
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
async def upload_image_manually():
    global result_image_buffer, origin_image_buffer
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        filelist = await request.files
        if 'file' not in filelist:
            return redirect(request.url)

        file = filelist['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_image(file.filename):
            # The image file seems valid! Detect faces and return the result.
            result = recognition.recognize_faces_in_image(file)
            img = add_face_rectangle(result, file)
            retval, buff = cv2.imencode('.jpg', img)
            result_image_buffer = buff.copy()

            ori_img = save_ori_image(file)
            retval, ori_buff = cv2.imencode('.jpg', ori_img)
            origin_image_buffer = ori_buff.copy()
            return await render_template('showimage.html')

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
async def get_result_image():
    global result_image_buffer
    frame = result_image_buffer.tobytes()
    return Response(frame, mimetype='image/jpg')

@app.route('/origin_image')
async def get_origin_image():
    global origin_image_buffer
    frame = origin_image_buffer.tobytes()
    return Response(frame, mimetype='image/jpg')


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)



@app.route('/capture', methods=['GET', 'POST'])
async def upload_image():
    import base64, uuid
    if request.method == 'GET':
        return await render_template('index.html')

    if request.method == 'POST' and 'photo' in request.form:
        filename = uuid.uuid1()
        if 'cec_id' in request.form and request.form['cec_id'] != '':
            filename = "{}{}".format(filename,request.form['cec_id'])
        upg_photo = request.form['photo'].split(',')[-1]
        file = os.path.join(APP_ROOT, "face_test/{}.png".format(filename))
        with open(file, "wb") as fh:
            fh.write(base64.b64decode(upg_photo))
        result = recognition.recognize_faces_in_image(file)
        faces = list(result['face_data'].values())
        faces.sort(key=lambda k:k['left'])
        if 'enable_face_plus' in request.form and request.form['enable_face_plus'] == 'on':
            face_datas = get_external_result(upg_photo)
            for i in range(len(faces)):
                face_datas[i]['name'] = faces[i]['name']
            return await render_template('result.html',face_datas = face_datas)

        return await render_template('result.html',face_datas = faces)

if __name__ == "__main__":
    face_db = os.path.join(APP_ROOT, "face_db")
    recognition = Face_Recognition()
    recognition.scan_known_people(face_db)
    app.run(host='0.0.0.0', port=5001, certfile='cert/cert.pem', keyfile='cert/key.pem')
