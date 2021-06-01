import numpy as np
import sys
import cv2


def load_annotation(path):
    raw_data = []
    with open(path, 'r') as f:
        raw_data = f.readlines()

    data = []  # official data
    for line in raw_data:
        name, bboxes = get_data(line)
        data.append({
            "name": name,
            "bboxes": bboxes
        })

    return data


def get_data(line):
    data = line.split(' ')
    name = data[0]
    bboxes = []
    data = data[1:]  # remov the name
    for box_str in data:
        box = list(map(float, box_str.split(',')))
        bboxes.append(box)  # exclude the id
    return name, np.array(bboxes)


def loading_bar(count, total):
    i = int(count/total * 50) + 1
    sys.stdout.write('\r')
    sys.stdout.write('[%-50s] %d%%' % ('='*i, i*2))
    sys.stdout.flush()


box_color = (0, 255, 255)
def display_box(img, boxes):
    for box in boxes:
        box = list(map(int, box))
        img = cv2.rectangle(img.copy(), (box[0], box[1]),
                            (box[2], box[3]), box_color, 2)
    return img


def resize_img(target_size, img, bboxes):
    x_scale = target_size[0] / img.shape[1]
    y_scale = target_size[1] / img.shape[0]

    new_bboxes = []
    for box in bboxes:
        x1, y1, x2, y2, class_id = box

        x1 = int(x1 * x_scale)
        x2 = int(x2 * x_scale)
        y1 = int(y1 * y_scale)
        y2 = int(y2 * y_scale)

        new_bboxes.append([x1, y1, x2, y2, class_id])

    return cv2.resize(img, target_size), np.array(new_bboxes)


def cutmix_loss(box1, box2, get_inter_box=False):
    if not is_intersect(box1, box2):
        if get_inter_box:
            return 0, None
        return 0

    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # area of the first box
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)

    if get_inter_box:
        return interArea / area1, np.array([x1, y1, x2, y2, box1[4]])
    return interArea / area1


def is_intersect(box1, box2):
    x1 = box1[0]
    y1 = box1[1]
    w1 = box1[2] - x1
    h1 = box1[3] - y1

    x2 = box2[0]
    y2 = box2[1]
    w2 = box2[2] - x2
    h2 = box2[3] - y2

    return x1 < x2 + w2 and x1 + w1 > x2\
        and y1 < y2 + h2 and y1 + h1 > y2


def get_random_mask(img_size):
    prob = np.random.random()
    div = 1/3

    if prob < div:
        # 2x2 mask
        mask1 = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        cell_width1 = int(img_size/4)
        cell1 = np.zeros((cell_width1, cell_width1, 3), dtype=np.uint8)
        for i in range(2):
            for j in range(2):
                row = 2 * i + 1
                col = 2 * j + 1
                mask1[row*cell_width1:(row+1)*cell_width1,
                      col*cell_width1:(col+1)*cell_width1] = cell1
        return mask1

    if prob < div * 2:
        # 3x3 mask
        mask2 = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        cell_width2 = int(img_size/5)
        cell2 = np.zeros((cell_width2, cell_width2, 3), dtype=np.uint8)
        for i in range(3):
            for j in range(3):
                row = 2 * i
                col = 2 * j
                mask2[row*cell_width2:(row+1)*cell_width2,
                      col*cell_width2:(col+1)*cell_width2] = cell2
        return mask2

    # 5x5 mask
    mask3 = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    cell_width3 = int(img_size/10)
    cell3 = np.zeros((cell_width3, cell_width3, 3), dtype=np.uint8)
    for i in range(5):
        for j in range(5):
            row = 2 * i + 1
            col = 2 * j + 1
            mask3[row*cell_width3:(row+1)*cell_width3,
                  col*cell_width3:(col+1)*cell_width3] = cell3
    return mask3
