#!/bin/bash

IMAGE=synergy-lite-face-recognition

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

VERSION=$1
LOCALNAME=$USER/$IMAGE:$VERSION

docker build \
    --build-arg "HTTP_PROXY=$http_proxy" \
    --build-arg "HTTPS_PROXY=$https_proxy" \
    --build-arg "FTP_PROXY=$ftp_proxy" \
    --build-arg "NO_PROXY=$no_proxy" \
    -t $LOCALNAME .
