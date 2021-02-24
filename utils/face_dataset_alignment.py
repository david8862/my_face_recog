#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, argparse
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from libs.face_net import Face_Recognition


def main():
    parser = argparse.ArgumentParser(description='Do face crop and align on raw face dataset to generate aligned faces')
    parser.add_argument('--input_path', help='Path for raw face dataset', type=str, required=True)
    parser.add_argument('--output_path', help='Output path for aligned face dataset', type=str, required=True)
    parser.add_argument('--face_size', help='Output face image size, default=%(default)s', type=int, required=False, default=128)
    args = parser.parse_args()

    class_names = os.listdir(args.input_path)
    recognition = Face_Recognition()
    os.makedirs(args.output_path, exist_ok=True)

    pbar = tqdm(total=len(class_names), desc='Processing face data')
    for file_path, file_dir, files in os.walk(args.input_path):
        for checked_file in files:
            class_name = file_path.split(os.path.sep)[-1]
            #print('checked_file', checked_file)
            #print('class_name', class_name)

            image_file = os.path.join(file_path, checked_file)
            aligned_face = recognition.load_aligned_face(image_file, face_size=args.face_size, margin=44)  #default margin 44

            if aligned_face is not None:
                os.makedirs(os.path.join(args.output_path, class_name), exist_ok=True)
                output_file = os.path.join(args.output_path, class_name, checked_file)
                Image.fromarray(aligned_face).save(output_file, quality=95)
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()

