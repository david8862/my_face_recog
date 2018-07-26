FROM ubuntu:16.04

MAINTAINER Xiaobin Zhang <xiaobizh@cisco.com>

# install tools
RUN apt update \
    && apt install -y iputils-ping net-tools wget curl vim git cmake imagemagick virtualenv python-pip python3-pip python-opencv python-dev libssl-dev libffi-dev build-essential

# Change workdir
WORKDIR /root

# build dlib & face_recognition env
RUN git clone https://github.com/davisking/dlib.git \
    && cd dlib \
	&& mkdir build; cd build; cmake .. -DDLIB_USE_CUDA=0 -DUSE_SSE4_INSTRUCTIONS=1 -DUSE_AVX_INSTRUCTIONS=0; cmake --build . \
    && cd .. \
    && python3 setup.py install --yes USE_SSE4_INSTRUCTIONS --no USE_AVX_INSTRUCTIONS --no DLIB_USE_CUDA \
    && pip3 install face_recognition opencv-python pillow flask requests pyOpenSSL

# build facenet env
RUN pip3 install tensorflow==1.7 scipy scikit-learn opencv-python h5py matplotlib Pillow requests psutil facenet

# Setup start up scripts
RUN mkdir /etc/dinit
COPY dinit/dinit /sbin/
COPY dinit/scripts /etc/dinit

CMD ["bash"]
ENTRYPOINT ["/sbin/dinit", "-c", "-s", "/etc/dinit" ]

RUN git clone https://github.com/david8862/my_face_recog.git


# Add container metadata
LABEL description="Develop docker for python face_recognition" version="0.2"
