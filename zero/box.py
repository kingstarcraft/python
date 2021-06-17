import numpy as np


def split_box(inputs, keep_dim=True):
    outputs = []
    for i in range(4):
        outputs.append(np.reshape(inputs[:, i], (-1, 1)) if keep_dim else inputs[:, i])
    return tuple(outputs)


def intersection(boxes1, boxes2):
    x_min1, y_min1, x_max1, y_max1 = split_box(boxes1)
    x_min2, y_min2, x_max2, y_max2 = split_box(boxes2)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape, np.float32),
        all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape, np.float32),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def area_box(inputs):
    x_min, y_min, x_max, y_max = split_box(inputs, keep_dim=False)
    return (y_max - y_min) * (x_max - x_min)


def iou(boxes1, boxes2):
    area1 = area_box(boxes1)
    area2 = area_box(boxes2)
    intersect = intersection(boxes1, boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / union


def ioa(boxes1, boxes2):
    area1 = area_box(boxes1)
    area2 = area_box(boxes2)
    intersect = intersection(boxes1, boxes2)
    union = np.minimum(np.expand_dims(area1, axis=1), np.expand_dims(area2, axis=0))
    return intersect / union


def filter_box(boxes, overlap=0.9):
    if len(boxes) == 0:
        return boxes
    score = boxes.shape[-1] == 5

    area = area_box(boxes)
    mask = np.ones(len(boxes), dtype='bool')
    for x, y in zip(*np.where(ioa(boxes, boxes) > overlap)):
        if x == y:
            continue
        if area[x] > area[y]:
            mask[y] = False
        elif area[x] == area[y] and score:
            mask[x if boxes[x][-1] < boxes[y][-1] else y] = False
        else:
            mask[x] = False
    return boxes[mask]

