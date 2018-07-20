#!/bin/bash

### Template launcher script for SL Docker
# Invoke with no arguments for an interactive shell, or pass a command
# to execute within the Docker container.
#
# You can set extra options for "docker run" with SLDOCKER_OPTIONS:


#
# Proxy Settings to access website outside of cisco
# export SLPROXY_OPTIONS=" \
#     -e  http_proxy=http://proxy-wsa.esl.cisco.com:80 \
#     -e  https_proxy=http://proxy-wsa.esl.cisco.com:80 \
#     -e  ftp_proxy=http://proxy-wsa.esl.cisco.com:80 \
#     -e  git_proxy=http://proxy-wsa.esl.cisco.com:80 \
#     -e  socks_server=http://proxy-wsa.esl.cisco.com:1080 \
#     -e  no_proxy='localhost,.localdomain,.cisco.com,.tandberg.com'"


###

### VARIABLES
HOST=containers.cisco.com
if $(docker info -f '{{.RegistryConfig.Mirrors}}' | grep '[http://10.74.130.230:5000/]' &>/dev/null); then
    DOCKER_IMAGE=xiaobizh/synergy-lite-face-recognition:latest
else
    DOCKER_IMAGE=${HOST}/xiaobizh/synergy-lite-face-recognition:latest
fi

## Pull latest from Docker registry
echo "Pulling $DOCKER_IMAGE..."
eval docker pull $DOCKER_IMAGE
if [ $? -ne 0 ]; then
    exit 1
fi

### Commands to Run
CMD=/root/sl-face-service/web_service.py

### Run the docker image
echo "Launching..."
eval docker run \
    -h "face-rec-develop" \
	-p 5001:5001 \
    --dns 64.104.123.245 \
    --dns 171.70.168.183 \
    -v /dev/null:/dev/null \
    -v /face_test:/root/sl-face-service/face_test \
    $SLPROXY_OPTIONS \
    --rm \
    -d \
    $DOCKER_IMAGE \
    $CMD
