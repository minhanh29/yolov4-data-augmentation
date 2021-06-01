import cv2
import numpy as np
import random
import sys

DATA_DIR = '../YOLO/dataset/img_for_mosaic'
ANNO_DIR = '../YOLO/dataset/anno_for_mosaic'
TARGET_DIR = 'img'
TARGET_ANNO = 'anno'


def load_annotation(path):
    data_arr = []
    with open(path, 'r') as f:
        data_arr = f.readlines()

    return data_arr


def get_data(line):
    data = line.split(' ')
    name = data[0]
    boxes = []
    data = data[1:]  # remov the name
    for box_str in data:
        box = list(map(float, box_str.split(',')))
        boxes.append(box)  # exclude the id

    return name, np.array(boxes)


data_arr = load_annotation(f"{ANNO_DIR}/voc_annotations.txt")
random.shuffle(data_arr)
count = 0
total = 30
with open(f"{TARGET_ANNO}/sample_annotation.txt", 'w') as f:
    for line in data_arr:
        name, boxes = get_data(line)
        img = cv2.imread(f"{DATA_DIR}/{name}")

        cv2.imwrite(f"{TARGET_DIR}/{name}", img)
        f.write(f"{line}")

        count += 1
        i = int(count/total * 50)
        sys.stdout.write('\r')
        sys.stdout.write('[%-50s] %d%%' % ('='*i, i*2))
        sys.stdout.flush()

        if count >= total:
            break

