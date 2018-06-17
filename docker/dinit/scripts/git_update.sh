#!/bin/bash
pushd /home/xiaobizh/my-face-recog 2>&1 >/dev/null
[ -d .git ] && git pull -q
popd 2>&1 >/dev/null
