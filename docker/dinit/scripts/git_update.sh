#!/bin/bash
pushd /root/my_face_recog 2>&1 >/dev/null
[ -d .git ] && git pull -q
popd 2>&1 >/dev/null
