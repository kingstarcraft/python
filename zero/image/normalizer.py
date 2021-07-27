import numpy as np
import spams
import cv2
from . import conversion


class ReinhardNormal(object):
    def __init__(self, from_color, to_color, target=None):
        self._from_color = from_color
        self._to_color = to_color
        self._target = target

    def __call__(self, inputs, dst=None, src=None, offset=None):
        outputs = self._from_color(inputs)
        if src is None:
            shape = outputs.shape
            temps = np.reshape(outputs, [-1, shape[-1]])
            src = np.mean(temps, axis=0), np.std(temps, axis=0)
        else:
            src = np.array(src[0]), np.array(src[1])
        if offset is not None:
            src = src[0] + offset[0], src[1] + offset[1]
        if dst is None:
            dst = self._target
        dst = np.array(dst[0]), np.array(dst[1])
        outputs = (outputs - src[0]) / src[1]
        outputs = outputs * dst[1] + dst[0]
        return self._to_color(outputs)


class ReinhardNormalBGR(ReinhardNormal):
    def __init__(self, target=None):
        super(ReinhardNormalBGR, self).__init__(conversion.BGR2LAB(), conversion.LAB2BGR(), target)


class ReinhardNormalRGB(ReinhardNormal):
    def __init__(self, target=None):
        super(ReinhardNormalRGB, self).__init__(conversion.RGB2LAB(), conversion.LAB2RGB(), target)


class BrightnessNormal(object):
    def __init__(self, ratio=90):
        self._ratio = ratio

    def __call__(self, inputs):
        percentile = np.percentile(inputs, self._ratio)
        return np.clip(inputs * 255.0 / percentile, 0, 255).astype(np.uint8)


class VahadaneNormal(object):
    def __init__(self, target=None, brightness=None):
        if brightness is not None:
            self._brightness = BrightnessNormal(brightness)
        if target is not None:
            self._stain = self.stain(
                target if brightness is None else self._brightness(target)
            ) if len(target.shape) == 3 else target

    def __call__(self, inputs, dst=None, src=None, brightness=90):
        brightness = self._brightness if brightness is None else BrightnessNormal(brightness)
        inputs = brightness(inputs)
        concentrations = self.lasso(inputs, self.stain(inputs) if src is None else src)
        return 255 * np.exp(-1 * np.dot(concentrations, self._stain if dst is None else dst).reshape(inputs.shape))

    def __array__(self):
        return self._stain

    def gray_mask(self, image, threshold):
        raise NotImplementedError

    def stain(self, image, threshold=0.8, lamda=0.1):
        """
        Get 2x3 stain matrix. First row H and second row E
        :param image:
        :param threshold:
        :param lamda:
        :return:
        """

        mask = self.gray_mask(image, threshold).reshape((-1,))
        od = conversion.COLOR2OD()(image).reshape((-1, 3))[mask]
        dictionary = spams.trainDL(
            od.T, K=2, lambda1=lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False
        ).T
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]
        return dictionary / np.linalg.norm(dictionary, axis=1)[:, None]

    def lasso(self, image, stain, lamda=0.01):
        od = conversion.COLOR2OD()(image).reshape((-1, 3))
        return spams.lasso(od.T, D=stain.T, mode=2, lambda1=lamda, pos=True).toarray().T


class VahadaneNormalRGB(VahadaneNormal):
    def gray_mask(self, image, threshold):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return lab[:, :, 0] / 255.0 < threshold


class VahadaneNormalBGR(VahadaneNormal):
    def gray_mask(self, image, threshold):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0] / 255.0 < threshold
