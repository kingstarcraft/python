import numpy as np
from sklearn.neighbors import KDTree
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


class NMS(object):
    def __init__(self, type='iou', threshold=0.5, index=False):
        self._nms = self.iou if type == 'iou' else (self.dynamic if threshold is None else self.distance)
        self.threshold = threshold
        self._index = index

    def iou(self, boxes):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        order = boxes[:, 4].argsort()[::-1]
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
            index = np.where(over <= self.threshold)[0]
            order = order[index + 1]
        if self._index:
            return keep
        else:
            return boxes[keep]

    def dynamic(self, boxes):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        order = boxes[:, 4].argsort()[::-1]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        width = (x2 - x1 + 1)
        height = (y2 - y1 + 1)
        size = np.sqrt(width ** 2 + height ** 2) / 2
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            center_x1 = center_x[i]
            center_y1 = center_y[i]
            center_x2 = center_x[order[1:]]
            center_y2 = center_y[order[1:]]
            dist = np.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)
            distances = np.maximum(size[order[1:]], size[i])
            index = np.where(dist > distances)[0]
            order = order[index + 1]
        if self._index:
            return keep
        else:
            return boxes[keep]

    def distance(self, boxes):
        scores = boxes[:, -1]
        if boxes.shape[-1] >= 4:
            center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
            center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2
        else:
            center_x = boxes[:, 0]
            center_y = boxes[:, 1]

        X = np.dstack((center_x, center_y))[0]
        tree = KDTree(X)

        sorted_ids = np.argsort(scores)[::-1]

        keep = []
        ind = tree.query_radius(X, r=self.threshold)

        while len(sorted_ids) > 0:
            ids = sorted_ids[0]
            keep.append(ids)
            sorted_ids = np.delete(sorted_ids, np.in1d(sorted_ids, ind[ids]).nonzero()[0])
        if self._index:
            return keep
        else:
            return boxes[keep]

    def __call__(self, boxes):
        return self._nms(boxes)


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
