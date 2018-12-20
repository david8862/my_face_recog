FROM ubuntu:16.04

MAINTAINER Xiaobin Zhang <xiaobizh@cisco.com>

# Change workdir
WORKDIR /root

# install tools
RUN apt update \
    && apt install -y iputils-ping \
    net-tools \
    wget \
    curl \
    vim \
    git \
    cmake \
    build-essential \
    checkinstall \
    imagemagick \
    virtualenv \
    python-pip \
    python3-pip \
    python-opencv \
    python-dev \
    python3-dev \
    python3-setuptools \
    libx11-dev \
    libssl-dev \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libsqlite3-dev \
    libffi-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    zlib1g-dev \
    openssl \
    libgtk-3-0 \
    libdbus-glib-1-2 \
    && curl -LO https://github.com/mozilla/geckodriver/releases/download/v0.21.0/geckodriver-v0.21.0-linux64.tar.gz \
    && tar -xf geckodriver-v0.21.0-linux64.tar.gz -C /sbin \
    && rm -rf geckodriver-v0.21.0-linux64.tar.gz

# Install python 3.7
RUN mkdir /tmp/Python37 \
    && cd /tmp/Python37 \
    && wget https://www.python.org/ftp/python/3.7.1/Python-3.7.1.tar.xz \
    && tar xvf Python-3.7.1.tar.xz \
    && cd /tmp/Python37/Python-3.7.1 \
    && ./configure --enable-optimizations \
    && make altinstall

# Install firefox 61.0.1
RUN cd /usr/local \
    && wget http://ftp.mozilla.org/pub/firefox/releases/61.0.1/linux-x86_64/en-US/firefox-61.0.1.tar.bz2 \
    && tar xvjf firefox-61.0.1.tar.bz2 \
    && ln -s /usr/local/firefox/firefox /usr/bin/firefox

# build dlib & face_recognition env
RUN git clone https://github.com/davisking/dlib.git \
    && cd dlib \
    && mkdir build; cd build; cmake .. -DDLIB_USE_CUDA=0 -DUSE_SSE4_INSTRUCTIONS=1 -DUSE_AVX_INSTRUCTIONS=1; cmake --build . \
    && cd .. \
    && python3.7 setup.py install --yes USE_SSE4_INSTRUCTIONS --yes USE_AVX_INSTRUCTIONS --no DLIB_USE_CUDA \
    && pip3.7 install face_recognition opencv-python pillow scikit-image quart flask gunicorn flask_monitoringdashboard requests pyOpenSSL selenium

# build facenet & openface env
RUN git clone https://github.com/cmusatyalab/openface.git \
    && cd openface \
    && python3.7 setup.py install \
    && cd /root/ \
    && wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl \
    && mv tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl tensorflow-1.12.0-cp37-cp37m-linux_x86_64.whl \
    && pip3.7 install tensorflow-1.12.0-cp37-cp37m-linux_x86_64.whl \
    && pip3.7 install scipy scikit-learn opencv-python h5py matplotlib Pillow requests psutil facenet

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
