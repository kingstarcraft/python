import torch

from zero.torch.data import conversion, normalizer, transforms
import numpy as np
import cv2


def draw_box(input, boxes):
    if isinstance(input, torch.Tensor):
        input = input.cpu().numpy()
    output = np.clip(input, 0, 255).astype('uint8').copy()
    shape = input.shape
    for box in boxes:
        y_min = int(box[1] * shape[0] + 0.5)
        x_min = int(box[2] * shape[1] + 0.5)
        y_max = int(box[3] * shape[0] + 0.5)
        x_max = int(box[4] * shape[1] + 0.5)
        cv2.circle(output, (x_min, y_min), 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(output, (x_max, y_max), 3, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1, cv2.LINE_AA)
    return output


src_image = cv2.imread('E:/11/src.bmp')
shape = src_image.shape
src_boxes = np.array([
    [1, 263 / shape[0], 96 / shape[1], 319 / shape[0], 722 / shape[1]],
    [2, 452 / shape[0], 286 / shape[1], 512 / shape[0], 529 / shape[1]]
])
rotate = transforms.RandomRotate()
filp = transforms.RandomFilp('vertical')
gaussian = transforms.GaussianNoise()
blur = transforms.Blur()
torch_normlizer = normalizer.ReinhardNormalBGR(
    ((2, 4, 5), (1, 3, 4)), ((3, 2, 1), (4, 3, 1)),
)
import zero.image.normalizer as norm

numpy_normalizer = norm.ReinhardNormalBGR()

cv2.imshow("src", draw_box(src_image, src_boxes))

func = blur
import time

while True:
    # list of numpy
    start = time.time()
    dst_images, dst_boxes = func([torch.Tensor(src_image)], [torch.Tensor(src_boxes)])
    print(time.time() - start)

    src_numpy = np.arange(0, 60).reshape(5, 4, 3).astype('float32')
    dst_numpy = np.clip(numpy_normalizer(src_numpy, ((3, 2, 1), (4, 3, 1)), ((2, 4, 5), (1, 3, 4))), 0, 255).astype('uint8')
    src_tensor = torch.Tensor(src_numpy)
    dst = np.clip(torch_normlizer(src_tensor).cpu().numpy(), 0, 255).astype('uint8')
    print(dst)
    cv2.imshow("dst.png", draw_box(dst_images[0], dst_boxes[0]))
    cv2.waitKey()
