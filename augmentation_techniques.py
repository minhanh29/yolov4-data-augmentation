import cv2
import numpy as np
import random
from utils import resize_img, cutmix_loss, get_random_mask
from config import DATA_DIR, IMAGE_SIZE


def mosaic(images, bboxes_list, num_box_threshod=1):
    """ Mosaic Data Augmentation

    Parameters
    ----------
    images: numpy.ndarray
        list of 4 images (opencv format)
    bboxes_list: numpy.ndarray
        list of 4 bboxes corresponding to the input iamges.
        Each box is an numpy.ndarray of the shape
        (xmin, ymin, xmax, ymax, class_id)
    num_box_threshod: int
        minimum number of boxes must be available in the combined image

    Returns
    -------
    image: numpy.ndarray
        combined image
    bboxes: numpy.ndarray
        new bboxes for the combined image

    if there are not enough bounding boxes that satisfy the num_box_threshod,
    return None, None
    """
    if len(images) != 4 or len(bboxes_list) != 4:
        print("Image and bboxes list must have a length of 4")
        return None, None

    # choose a dividing point
    div_point = (int(IMAGE_SIZE/2), IMAGE_SIZE - int(IMAGE_SIZE/2))

    # size of each image
    top_left_size = div_point
    top_right_size = (IMAGE_SIZE - div_point[0], div_point[1])
    bottom_left_size = (div_point[0], IMAGE_SIZE - div_point[1])
    bottom_right_size = (IMAGE_SIZE - div_point[0], IMAGE_SIZE - div_point[1])
    new_sizes = [top_left_size, top_right_size,
                 bottom_left_size, bottom_right_size]

    # translate vector
    translate_vectors = [
        (0, 0),  # top left
        (div_point[0], 0),  # top right
        (0, div_point[1]),  # bottom left
        (div_point[0], div_point[1])  # bottom right
    ]

    # base image
    base_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    # base_img, b = resize_img((IMAGE_SIZE, IMAGE_SIZE), img, [])
    bboxes = []

    # resize all images
    index = 0
    for target_size, img, m_bbox, translate_vector in\
            zip(new_sizes, images, bboxes_list, translate_vectors):
        res_img, new_bboxes = resize_img(tuple(reversed(target_size)),
                                         img, m_bbox)

        if index == 0:
            base_img[:div_point[0], :div_point[1]] = res_img[:, :]
        elif index == 1:
            base_img[div_point[0]:, :div_point[1]] = res_img[:, :]
        elif index == 2:
            base_img[:div_point[0], div_point[1]:] = res_img[:, :]
        else:
            base_img[div_point[0]:, div_point[1]:] = res_img[:, :]
        index += 1

        for box in new_bboxes:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            class_id = box[4]

            bboxes.append([x1 + translate_vector[1], y1 + translate_vector[0],
                           x2 + translate_vector[1], y2 + translate_vector[0],
                           class_id])

    # randomly crop image
    c_x1 = random.randint(0, int(IMAGE_SIZE * 0.4))
    c_y1 = random.randint(0, int(IMAGE_SIZE * 0.4))
    c_w = int(IMAGE_SIZE * 0.6)
    c_x2 = c_x1 + c_w
    c_y2 = c_y1 + c_w

    crop_img = base_img[c_y1:c_y2, c_x1:c_x2]
    crop_translate_vector = (-c_x1, -c_y1)
    result_bboxes = []
    crop_img_box = (c_x1, c_y1, c_x2, c_y2)
    for box in bboxes:
        overlap, inter_box = cutmix_loss(box, crop_img_box, get_inter_box=True)
        if overlap is None or overlap < 0.85:
            continue
        x1 = inter_box[0]
        y1 = inter_box[1]
        x2 = inter_box[2]
        y2 = inter_box[3]
        class_id = box[4]

        result_bboxes.append([x1 + crop_translate_vector[0],
                              y1 + crop_translate_vector[1],
                              x2 + crop_translate_vector[0],
                              y2 + crop_translate_vector[1],
                              class_id])

    if len(result_bboxes) < num_box_threshod:
        return None, None

    return resize_img((IMAGE_SIZE, IMAGE_SIZE), crop_img, result_bboxes)


def mix_up(images_list, bboxes_list, num_box_threshod=1):
    """ Mix Up Data Augmentation

    Parameters
    ----------
    images: numpy.ndarray
        list of 2 images (opencv format)
    bboxes_list: numpy.ndarray
        list of 2 bboxes corresponding to the input iamges.
        Each box is an numpy.ndarray of the shape
        (xmin, ymin, xmax, ymax, class_id)
    num_box_threshod: int
        minimum number of boxes must be available in the combined image

    Returns
    -------
    image: numpy.ndarray
        combined image
    bboxes: numpy.ndarray
        new bboxes for the combined image

    if there are not enough bounding boxes that satisfy the num_box_threshod,
    return None, None
    """

    if len(images_list) != 2 or len(bboxes_list) != 2:
        print("Image and bboxes list must have a length of 2")
        return None, None

    new_bboxes = None
    images = []
    for img, bboxes in zip(images_list, bboxes_list):
        img, bboxes = resize_img((IMAGE_SIZE, IMAGE_SIZE),
                                 img.copy(), bboxes.copy())
        if new_bboxes is None:
            new_bboxes = bboxes
        else:
            new_bboxes = np.vstack([new_bboxes, bboxes])
        images.append(img)

    alpha = 0.5
    mixup_img = np.uint8(images[0] * alpha + images[1] * (1 - alpha))

    if len(new_bboxes) < num_box_threshod:
        return None, None

    return mixup_img, new_bboxes


def cut_mix(images, bboxes_list, num_box_threshod=1):
    """ Cut Mix Data Augmentation

    Parameters
    ----------
    images: numpy.ndarray
        list of 2 images (opencv format)
    bboxes_list: numpy.ndarray
        list of 2 bboxes corresponding to the input iamges.
        Each box is an numpy.ndarray of the shape
        (xmin, ymin, xmax, ymax, class_id)
    num_box_threshod: int
        minimum number of boxes must be available in the combined image

    Returns
    -------
    image: numpy.ndarray
        combined image
    bboxes: numpy.ndarray
        new bboxes for the combined image

    if there are not enough bounding boxes that satisfy the num_box_threshod,
    return None, None
    """

    if len(images) != 2 or len(bboxes_list) != 2:
        print("Image and bboxes list must have a length of 2")
        return None, None

    # width and height of the cutting box
    b_width = int(0.4 * IMAGE_SIZE)
    b_height = int(0.6 * IMAGE_SIZE)

    # define the upper left corner
    b_x1 = random.randint(0, IMAGE_SIZE - b_width)
    b_y1 = random.randint(0, IMAGE_SIZE - b_height)

    # bottom right corner
    b_x2 = b_x1 + b_width
    b_y2 = b_y1 + b_height

    # base image
    base_img = images[0]
    base_bboxes = bboxes_list[0]

    # second image
    second_img = images[1]
    second_bboxes = bboxes_list[1]

    # resize the images
    base_img, base_bboxes = resize_img((IMAGE_SIZE, IMAGE_SIZE),
                                       base_img.copy(),
                                       base_bboxes.copy())
    second_img, second_bboxes = resize_img((IMAGE_SIZE, IMAGE_SIZE),
                                           second_img.copy(),
                                           second_bboxes.copy())

    # cut the second image (h, w, c)
    overlap = second_img[b_y1:b_y2, b_x1:b_x2]

    # paste ovelap to the base image
    base_img[b_y1:b_y2, b_x1:b_x2] = overlap

    # remove base boxes that are covered by the second image
    img_box = (b_x1, b_y1, b_x2, b_y2)
    base_bboxes = np.array([box for box in base_bboxes
                            if cutmix_loss(box, img_box) < 0.5])

    # remove cut bboxes of the second image
    filter_second_boxes = []
    for box in second_bboxes:
        gain, inter_box = cutmix_loss(box, img_box, get_inter_box=True)
        if gain > 0.5:
            filter_second_boxes.append(inter_box)

    # combine bboxes
    bboxes = np.empty((0, 5))
    if len(base_bboxes) > 0:
        bboxes = np.vstack([bboxes, base_bboxes])

    if len(filter_second_boxes) > 0:
        bboxes = np.vstack([bboxes, filter_second_boxes])

    if (len(bboxes) == 0):
        return None, None
    return base_img, np.array(bboxes)


def grid_mask(img, bboxes):
    """ Grid Mask Data Augmentation

    Parameters
    ----------
    image: numpy.ndarray
    bboxes: numpy.ndarray
        List of bounding boxes in the image.
        Each box is an numpy.ndarray of the shape
        (xmin, ymin, xmax, ymax, class_id)

    Returns
    -------
    image: numpy.ndarray
        Image with a random grid mask applied
    bboxes: numpy.ndarray
        new bboxes for the combined image
    """

    img, bboxes = resize_img((IMAGE_SIZE, IMAGE_SIZE),
                             img.copy(), bboxes.copy())
    mask = get_random_mask(IMAGE_SIZE)
    return cv2.bitwise_and(img, mask), bboxes
