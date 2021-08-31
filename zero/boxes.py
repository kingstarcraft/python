import numpy as np
from . import box


def splits(inputs, keep_dim=True):
    outputs = []
    for i in range(4):
        outputs.append(np.reshape(inputs[:, i], (-1, 1)) if keep_dim else inputs[:, i])
    return tuple(outputs)


def intersection(boxes1, boxes2):
    x_min1, y_min1, x_max1, y_max1 = splits(boxes1)
    x_min2, y_min2, x_max2, y_max2 = splits(boxes2)

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


def area(inputs):
    x_min, y_min, x_max, y_max = splits(inputs, keep_dim=False)
    return (y_max - y_min) * (x_max - x_min)


def iou(boxes1, boxes2):
    area1 = area(boxes1)
    area2 = area(boxes2)
    intersect = intersection(boxes1, boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / union.clip(min=1e-10)


def ioa(boxes1, boxes2):
    area1 = area(boxes1)
    area2 = area(boxes2)
    intersect = intersection(boxes1, boxes2)
    union = np.minimum(np.expand_dims(area1, axis=1), np.expand_dims(area2, axis=0))
    return intersect / union.clip(min=1e-10)


def nms(dets, thresh):
    """
    nms
    :param dets: ndarray [x1,y1,x2,y2,score]
    :param thresh: int
    :return: list[index]
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    order = dets[:, 4].argsort()[::-1]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        over = (w * h) / (area[i] + area[order[1:]] - w * h)
        index = np.where(over <= thresh)[0]
        order = order[index + 1]
    return keep


class Merge(object):
    def __init__(self, similarity='iou', threshold=0.5, method='union'):
        '''
        :param threshold: [0-1]
        :param method: union/intersection
        '''
        self._threshold = threshold
        if similarity == 'iou':
            self._similarity = box.iou
        elif similarity == 'ioa':
            self._similarity = box.ioa
        else:
            raise NotImplementedError
        if method == 'intersection':
            self._core = box.intersection
        elif method == 'union':
            self._core = box.union
        else:
            raise NotImplementedError

    def __call__(self, boxes):
        boxes = np.array(boxes)
        change = True
        while change:
            cache = []
            change = False
            for i in range(len(boxes)):
                s = boxes[i]
                save = True
                for j in range(i + 1, len(boxes)):
                    d = boxes[j]
                    if self._similarity(s, d) > self._threshold:
                        save = False
                        change = True
                        merge = self._core(s, d)
                        if len(boxes[j]) > len(merge):
                            boxes[j][0:len(merge)] = merge
                            boxes[j][-1] = max(s[-1], d[-1])
                        else:
                            boxes[j] = merge
                if save:
                    cache.append(s)
            boxes = cache
        return boxes


class Filter(object):
    def __init__(self, similarity='ioa', threshold=0.9, priority='score'):
        self._threshold = threshold
        if similarity == 'iou':
            self._similarity = iou
        elif similarity == 'ioa':
            self._similarity = ioa
        else:
            raise NotImplementedError
        self._priority = priority

    def __call__(self, boxes):
        boxes = np.array(boxes)
        if len(boxes) == 0:
            return boxes
        score = boxes.shape[-1] == 5
        a = area(boxes)
        mask = np.ones(len(boxes), dtype='bool')
        if self._priority == 'score':
            for x, y in zip(*np.where(self._similarity(boxes, boxes) > self._threshold)):
                if x != y:
                    if boxes[y][-1] < boxes[x][-1]:
                        mask[y] = False
                    elif boxes[x][-1] == boxes[y][-1] and score:
                        mask[x if a[x] < a[y] else y] = False
                    else:
                        mask[x] = False
        else:
            for x, y in zip(*np.where(self._similarity(boxes, boxes) > self._threshold)):
                if x != y:
                    if a[y] < a[x]:
                        mask[y] = False
                    elif a[x] == a[y] and score:
                        mask[x if boxes[x][-1] < boxes[y][-1] else y] = False
                    else:
                        mask[x] = False
        return boxes[mask]
