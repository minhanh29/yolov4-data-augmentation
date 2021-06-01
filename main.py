import cv2
import numpy as np
import random
from config import DATA_DIR, ANNO_DIR, TARGET_DIR, TARGET_ANNO, TOTAL_SAMPLES
from augmentation_techniques import mosaic, mix_up, cut_mix, grid_mask
from utils import load_annotation, display_box, loading_bar


show_image = True
write_img = True


def write_anno(f, filename, img, bboxes):
    output = f"{filename}.jpg"
    cv2.imwrite(f"{TARGET_DIR}/{filename}.jpg", img)
    for box in bboxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        class_id = int(box[4])
        output += f" {x1},{y1},{x2},{y2},{class_id}"
    f.write(f"{output}\n")


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
    # SAMPLES = int(TOTAL_SAMPLES / 4)
    SAMPLES = 5

    if write_img:
        f = open(f"{TARGET_ANNO}/augmented_annotaion.txt", 'w')

    # mosaic
    count = 0
    print("Mosaic is in process")
    while True:
        # choose 4 images
        chosen_images = np.random.choice(data, 4, replace=False)
        images, bboxes_list = unpack_data(chosen_images)
        img, bboxes = mosaic(images, bboxes_list, 3)

        if img is None or bboxes is None:
            # print("Not enough boxes")
            continue

        result_img = display_box(img, bboxes)

        if show_image:
            cv2.imshow("Img", result_img)
            cv2.waitKey(500)

        if write_img:
            filename = f"Mosaic_{count}"
            write_anno(f, filename, img, bboxes)
            loading_bar(count, SAMPLES)

        count += 1
        if count >= SAMPLES:
            break

    # mix up
    count = 0
    print("\nMix up is in process")
    while True:
        # choose 4 images
        chosen_images = np.random.choice(data, 2, replace=False)
        images, bboxes_list = unpack_data(chosen_images)
        img, bboxes = mix_up(images, bboxes_list, 1)

        if img is None or bboxes is None:
            # print("Not enough boxes")
            continue

        result_img = display_box(img, bboxes)

        if show_image:
            cv2.imshow("Img", result_img)
            cv2.waitKey(500)

        if write_img:
            filename = f"Mixup_{count}"
            write_anno(f, filename, img, bboxes)
            loading_bar(count, SAMPLES)

        count += 1
        if count >= SAMPLES:
            break

    # cut mix
    count = 0
    print("\nCut mix is in process")
    while True:
        # choose 4 images
        chosen_images = np.random.choice(data, 2, replace=False)
        images, bboxes_list = unpack_data(chosen_images)
        img, bboxes = cut_mix(images, bboxes_list, 1)

        if img is None or bboxes is None:
            # print("Not enough boxes")
            continue

        result_img = display_box(img, bboxes)

        if show_image:
            cv2.imshow("Img", result_img)
            cv2.waitKey(500)

        if write_img:
            filename = f"Cutmix_{count}"
            write_anno(f, filename, img, bboxes)
            loading_bar(count, SAMPLES)

        count += 1
        if count >= SAMPLES:
            break

    # grid mask
    count = 0
    print("\nGrid mask is in process")
    while True:
        # choose 4 images
        chosen_images = np.random.choice(data, 1, replace=False)
        images, bboxes_list = unpack_data(chosen_images)
        img, bboxes = grid_mask(images[0], bboxes_list[0])

        if img is None or bboxes is None:
            # print("Not enough boxes")
            continue

        result_img = display_box(img, bboxes)

        if show_image:
            cv2.imshow("Img", result_img)
            cv2.waitKey(500)

        if write_img:
            filename = f"GridMask_{count}"
            write_anno(f, filename, img, bboxes)
            loading_bar(count, SAMPLES)

        count += 1
        if count >= SAMPLES:
            break

    f.close()


main()
