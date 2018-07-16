FROM ubuntu:16.04

MAINTAINER Xiaobin Zhang <xiaobizh@cisco.com>

# install tools
RUN apt update \
    && apt install -y iputils-ping vim git cmake imagemagick virtualenv python-pip python-opencv python-dev build-essential \
	&& pip install --upgrade pip

# Change workdir
WORKDIR /root

# create user
RUN git clone https://github.com/davisking/dlib.git \
    && cd dlib \
	&& mkdir build; cd build; cmake .. -DDLIB_USE_CUDA=0 -DUSE_SSE4_INSTRUCTIONS=1 -DUSE_AVX_INSTRUCTIONS=0; cmake --build . \
    && cd .. \
    && python setup.py install --yes USE_SSE4_INSTRUCTIONS --no USE_AVX_INSTRUCTIONS --no DLIB_USE_CUDA \
    && pip install face_recognition opencv-python flask pyOpenSSL

# Setup start up scripts
RUN mkdir /etc/dinit
COPY dinit/dinit /sbin/
COPY dinit/scripts /etc/dinit

CMD ["bash"]
ENTRYPOINT ["/sbin/dinit", "-c", "-s", "/etc/dinit" ]

# Change workdir to user home
WORKDIR /home/xiaobizh
RUN git clone https://xiaobizh@bitbucket-eng-sjc1.cisco.com/bitbucket/scm/~xiaobizh/my-face-recog.git


# Add container metadata
LABEL description="Develop docker for python face_recognition" version="0.2"
