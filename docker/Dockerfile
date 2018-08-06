FROM ubuntu:16.04

MAINTAINER Xiaobin Zhang <xiaobizh@cisco.com>

# Change workdir
WORKDIR /root

# install tools
RUN apt update \
    && apt install -y iputils-ping net-tools wget curl vim git cmake imagemagick virtualenv python-pip python3-pip python-opencv python-dev libssl-dev libffi-dev build-essential libgtk-3-0 libdbus-glib-1-2 \
    && curl -LO https://github.com/mozilla/geckodriver/releases/download/v0.21.0/geckodriver-v0.21.0-linux64.tar.gz \
    && tar -xf geckodriver-v0.21.0-linux64.tar.gz -C /sbin \
    && rm -rf geckodriver-v0.21.0-linux64.tar.gz

# Install firefox 61.0.1
RUN cd /usr/local \
    && wget http://ftp.mozilla.org/pub/firefox/releases/61.0.1/linux-x86_64/en-US/firefox-61.0.1.tar.bz2 \
    && tar xvjf firefox-61.0.1.tar.bz2 \
    && ln -s /usr/local/firefox/firefox /usr/bin/firefox

# build dlib & face_recognition env
RUN git clone https://github.com/davisking/dlib.git \
    && cd dlib \
	&& mkdir build; cd build; cmake .. -DDLIB_USE_CUDA=0 -DUSE_SSE4_INSTRUCTIONS=1 -DUSE_AVX_INSTRUCTIONS=0; cmake --build . \
    && cd .. \
    && python3 setup.py install --yes USE_SSE4_INSTRUCTIONS --no USE_AVX_INSTRUCTIONS --no DLIB_USE_CUDA \
    && pip3 install face_recognition opencv-python pillow flask requests pyOpenSSL selenium

# build facenet & openface env
RUN git clone https://github.com/cmusatyalab/openface.git \
    && cd openface \
    && python3 setup.py install \
    && pip3 install tensorflow==1.7 scipy scikit-learn opencv-python h5py matplotlib Pillow requests psutil facenet

# Setup start up scripts
RUN mkdir /etc/dinit
COPY dinit/dinit /sbin/
COPY dinit/scripts /etc/dinit

CMD ["bash"]
ENTRYPOINT ["/sbin/dinit", "-c", "-s", "/etc/dinit" ]

RUN git clone https://github.com/david8862/my_face_recog.git \
    && cd my_face_recog/libs/models/ \
    && wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
    && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Add container metadata
LABEL description="Develop docker for python face_recognition" version="0.2"
