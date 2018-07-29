## Python Face Recognition framework based on face_recognition

A secure web service for high-quality face recognition and some sample utility based on python "face_recognition" package, which is a good wrapper of dlib C++ library.

* web_service.py: web service for face recognition
* utils/my_face_detect.py: webcam live face detect & landmark display sample running on Macbook
* utils/my_face_recog.py: webcam live face recognition sample running on Macbook

Face recognition DB images could be store under face_db/. Web server & utils will automatically load from it.

### How to install & launch web service
```shell
pip3 install -r requirements.txt
pushd ..
git clone https://github.com/davisking/dlib.git
cd dlib; mkdir build; cd build; cmake .. -DDLIB_USE_CUDA=0 -DUSE_SSE4_INSTRUCTIONS=1 -DUSE_AVX_INSTRUCTIONS=0; cmake --build .
cd ..
python3 setup.py install --yes USE_SSE4_INSTRUCTIONS --no USE_AVX_INSTRUCTIONS --no DLIB_USE_CUDA
popd
python3 web_service.py
```
Basic service: https://localhost:5001/
WebRTC image capture tool: https://localhost:5001/capture

