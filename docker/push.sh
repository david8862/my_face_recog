#!/bin/bash

IMAGE=synergy-lite-face-recognition
SERVER=containers.cisco.com/xiaobizh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

VERSION=$1
LOCALNAME=$USER/$IMAGE:$VERSION
REMOTENAME=$SERVER/$IMAGE:$VERSION

docker tag $LOCALNAME $REMOTENAME && \
    docker push $REMOTENAME
