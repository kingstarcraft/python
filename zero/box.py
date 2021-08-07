def intersection(box1, box2):
    return max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])


def union(box1, box2):
    return min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])


def area(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(box1, box2):
    return area(intersection(box1, box2)) / area(union(box1, box2))


def ioa(box1, box2):
    return area(intersection(box1, box2)) / min(area(box1), area(box2))
