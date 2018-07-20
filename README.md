## Python Face Recognition framework based on face_recognition

Some samples for the python "face_recognition" package, which is a good wrapper of dlib C++ library.

* utils/my_face_detect.py: webcam live face detect & landmark display sample running on Macbook
* utils/my_face_recog.py: webcam live face recognition sample running on Macbook
* web_service.py: web service for face recognition based on flask

Face recognition DB under face_db/.
Docker implementation of web service under docker/ dir.

### How to launch web service
```shell
python web_service.py
```
