import random
import torch
from torchvision import transforms


def _sort_boxes(boxes):
    min_xy = torch.where(boxes[..., 1:3] < boxes[..., 3:5], boxes[..., 1:3], boxes[..., 3:5])
    max_xy = torch.where(boxes[..., 1:3] > boxes[..., 3:5], boxes[..., 1:3], boxes[..., 3:5])
    boxes[..., 1:3] = min_xy
    boxes[..., 3:5] = max_xy
    return boxes


class Transform(torch.nn.Module):
    def __init__(self):
        super(Transform, self).__init__()

    def _transform(self, inputs, params, func=None):
        batch_size = len(inputs)
        outputs = []
        for i in range(batch_size):
            outputs.append(func(inputs[i], None if params is None else params[i]))
        if isinstance(inputs, tuple):
            outputs = tuple(outputs)
        elif not isinstance(inputs, list):
            outputs = torch.stack(outputs)
        return outputs

    def _transform_images(self, input, param):
        '''
        :param input: [H,W,C]
        :return:
        '''
        return input

    def _transform_boxes(self, input, param):
        '''
        :param input: [N, [cls, y_min, x_min, y_max, x_max]]
        :return:
        '''
        return input

    def _transform_labels(self, input, param):
        return input

    def _sample(self, batch_size):
        return None

    @torch.no_grad()
    def __call__(self, images, boxes=None):
        '''
        :param image: a list of image tensor or a tensor with shape [B, H, W, C]
        :param boxes: a list of boxes tensor or a tensor with shape [B, N, [cls, y_min, x_min, y_max, x_max]]
        :return:
        '''
        results = []
        samples = self._sample(len(images))
        results.append(self._transform(images, samples, self._transform_images))
        if boxes is not None:
            results.append(self._transform(boxes, samples, self._transform_boxes))
        return tuple(results)


class ChoicesTransform(Transform):
    def __init__(self, transform, probability=None):
        super(ChoicesTransform, self).__init__()
        if isinstance(transform, int) or isinstance(transform, float):
            transform = (transform,)

        elif len(transform) == 2:
            if isinstance(probability, float) or isinstance(probability, int):
                probability = (1 - probability, probability)

        self._transforms = transform
        self._probability = tuple([1 / len(transform) for _ in transform]) if probability is None else probability
        assert len(self._probability) == len(self._transforms)

    def _sample(self, batch_size):
        return random.choices(self._transforms, self._probability, k=batch_size)


class RandomRotate(ChoicesTransform):
    def __init__(self, angles=(0, 90, 180, 270), probability=None):
        '''
        :param angles:
        '''
        super(RandomRotate, self).__init__(angles, probability)

    def _transform_images(self, input, param):
        output = torch.clone(input)
        return torch.rot90(output, param // 90)

    def _transform_boxes(self, input, param):
        '''
        :param inputs:  a tensor with shape [B, N, [cls, y_min, x_min, y_max, x_max]]
        :return:
        '''
        output = torch.clone(input)
        if param == 270 or param == -90:
            output[..., 1::2] = input[..., 2::2]  # y' = x
            output[..., 2::2] = 1 - input[..., 1::2]  # x' = 1 - y
        elif param == 180 or param == -180:
            output[..., 1::2] = 1 - input[..., 1::2]  # y' = 1 - y
            output[..., 2::2] = 1 - input[..., 2::2]  # x' = 1 - x
        elif param == 90 or param == -270:
            output[..., 1::2] = 1 - input[..., 2::2]  # y' = 1 - x
            output[..., 2::2] = input[..., 1::2]  # x' = y
        return _sort_boxes(output)


class RandomFilp(ChoicesTransform):
    def __init__(self, direction='horizontal', probability=None):
        if isinstance(direction, str):
            if direction == 'vertical':
                directions = (0, -3)
            elif direction == 'horizontal':
                directions = (0, -2)
            elif direction == 'diagonal':
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            directions = direction
        super(RandomFilp, self).__init__(directions, probability)

    def _transform_images(self, input, param):
        '''
        :param input: a tensor with shape[B,H,W,C]
        :param angle:
        :return:
        '''
        if param == 0:
            return torch.clone(input)
        return torch.flip(input, [param])

    def _transform_boxes(self, input, param):
        '''
        :param inputs:  a tensor with shape [B, N, [cls, y_min, x_min, y_max, x_max]]
        :return:
        '''
        output = torch.clone(input)
        if param == -3:
            output[..., 1::2] = 1 - input[..., 1::2]  # y' = 1 - y
            output[..., 2::2] = input[..., 2::2]  # x' = x
        elif param == -2:
            output[..., 1::2] = input[..., 1::2]  # y' = y
            output[..., 2::2] = 1 - input[..., 2::2]  # x' = 1 - x
        elif param > 0:
            raise NotImplementedError
        return _sort_boxes(output)


class GaussianNoise(ChoicesTransform):
    def __init__(self, stds=(0, 25, 50, 75), probability=(0.5, 0.25, 0.15, 0.1)):
        super(GaussianNoise, self).__init__(stds, probability)

    def _transform_images(self, input, param):
        noise = param * torch.randn_like(input)
        return input + noise


class Blur(ChoicesTransform):
    def __init__(self, sigma=(0.1, 0.2), kernel=(0, 3, 5), probability=None):
        core = {}
        for k in kernel:
            if k != 0:
                core[str(k)] = transforms.GaussianBlur(k, sigma)
        super(Blur, self).__init__(kernel, probability)
        self._core = torch.nn.ModuleDict(core)
        self._sigma = sigma

    def _transform_images(self, input, param):
        if param == 0:
            return torch.clone(input)
        return self._core[str(param)](input)
