#!/bin/bash
pushd /root/sl-face-service 2>&1 >/dev/null
[ -d .git ] && git pull -q
popd 2>&1 >/dev/null
