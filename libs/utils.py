#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png|gif|bmp)', f, flags=re.I)]

def allowed_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


