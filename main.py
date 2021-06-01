import cv2
import numpy as np
import random
from config import DATA_DIR, ANNO_DIR, TARGET_DIR, TARGET_ANNO, TOTAL_SAMPLES
from augmentation_techniques import mosaic, mix_up, cut_mix, grid_mask
from utils import load_annotation, display_box


def unpack_data(data):
    images = []
    bboxes_list = []
    for img_data in data:
        img = cv2.imread(f"{DATA_DIR}/{img_data['name']}")
        images.append(img)
        bboxes_list.append(img_data['bboxes'])

    return images, bboxes_list


def main():
    data = load_annotation(f"{ANNO_DIR}/sample_annotation.txt")
    # shuffle all the images
    random.shuffle(data)

    # SAMPLES = len(data) * 0.1
    SAMPLES = 5

    # mosaic
    count = 0
    while True:
        # choose 4 images
        chosen_images = np.random.choice(data, 4, replace=False)
        images, bboxes_list = unpack_data(chosen_images)
        img, bboxes = mosaic(images, bboxes_list, 3)

        if img is None or bboxes is None:
            print("Not enough boxes")
            continue

        result_img = display_box(img, bboxes)
        cv2.imshow("Img", result_img)
        cv2.waitKey(500)

        count += 1
        if count >= SAMPLES:
            break

    # mix up
    count = 0
    while True:
        # choose 4 images
        chosen_images = np.random.choice(data, 2, replace=False)
        images, bboxes_list = unpack_data(chosen_images)
        img, bboxes = mix_up(images, bboxes_list, 1)

        if img is None or bboxes is None:
            print("Not enough boxes")
            continue

        result_img = display_box(img, bboxes)
        cv2.imshow("Img", result_img)
        cv2.waitKey(500)

        count += 1
        if count >= SAMPLES:
            break

    # cut mix
    count = 0
    while True:
        # choose 4 images
        chosen_images = np.random.choice(data, 2, replace=False)
        images, bboxes_list = unpack_data(chosen_images)
        img, bboxes = cut_mix(images, bboxes_list, 1)

        if img is None or bboxes is None:
            print("Not enough boxes")
            continue

        result_img = display_box(img, bboxes)
        cv2.imshow("Img", result_img)
        cv2.waitKey(500)

        count += 1
        if count >= SAMPLES:
            break

    # grid mask
    count = 0
    while True:
        # choose 4 images
        chosen_images = np.random.choice(data, 1, replace=False)
        images, bboxes_list = unpack_data(chosen_images)
        img, bboxes = grid_mask(images[0], bboxes_list[0])

        if img is None or bboxes is None:
            print("Not enough boxes")
            continue

        result_img = display_box(img, bboxes)
        cv2.imshow("Img", result_img)
        cv2.waitKey(500)

        count += 1
        if count >= SAMPLES:
            break

    # with open(f"{TARGET_ANNO}/aug_annotations.txt", 'w') as f:
        # for count in range(TOTAL_SAMPLES):
        #     img = cv2.imread(f"{DATA_DIR}/{name}")

        #     cv2.imwrite(f"{TARGET_DIR}/{name}", img)

        #     loading_bar(count, TOTAL_SAMPLES)
        #     if count >= TOTAL_SAMPLES:
        #         break


main()
