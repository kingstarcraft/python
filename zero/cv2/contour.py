import cv2
import numpy as np


class Node(np.ndarray):
    def __new__(self, *args, **kwargs):
        obj = np.asarray(*args, **kwargs).view(self)
        obj.children = []
        obj.parent = None
        obj.label = None
        return obj


def relate(contours, hierarchy, address=False):
    size = len(contours)
    hierarchy = hierarchy.reshape((size, 4))
    nodes = [Node(contour) for contour in contours]

    for i in range(size):
        parent = hierarchy[i][-1]
        if parent != -1:
            if address:
                nodes[i].parent = parent
                nodes[parent].children.append(i)
            else:
                nodes[i].parent = nodes[parent]
                nodes[parent].children.append(nodes[i])
    return nodes


def tree(contours, hierarchy):
    nodes = relate(contours, hierarchy)
    ancestors = []
    for node in nodes:
        if node.parent is None:
            ancestors.append(node)
    return ancestors


def label(mask, contours, hierarchy, address=False):
    nodes = relate(contours, hierarchy, address)
    for node in nodes:
        roi = cv2.boundingRect(node)
        pos = np.array([[roi[0], roi[1]]])
        src = mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        dst = np.zeros_like(src)
        cv2.drawContours(dst, [node - pos], -1, 1, -1, cv2.LINE_AA)
        if len(node.children) > 0:
            if address:
                cv2.drawContours(dst, [(nodes[child] - pos) for child in node.children], -1, 0, -1, cv2.LINE_AA)
            else:
                cv2.drawContours(dst, [(child - pos) for child in node.children], -1, 0, -1, cv2.LINE_AA)
        node.label = int(np.mean(src[dst.astype('bool')]) + 0.5)
    return nodes
