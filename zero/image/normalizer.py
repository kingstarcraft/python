import numpy as np
import spams
import cv2
from . import conversion


class BrightnessNormal(object):
    def __init__(self, ratio=90):
        self._ratio = ratio

    def __call__(self, inputs):
        percentile = np.percentile(inputs, self._ratio)
        return np.clip(inputs * 255.0 / percentile, 0, 255).astype(np.uint8)


class Normal(object):
    def __init__(self, from_color, to_color, target=None, brightness=90, reverse=False):
        self._from = from_color
        self._to = to_color
        self._reverse = reverse
        if brightness is not None:
            self._brightness = BrightnessNormal(brightness)
        else:
            self._brightness = lambda images: images
        self._target = target
        if isinstance(target, np.ndarray):
            self._target = self.stain(target)

    def __call__(self, images, dst=None, src=None):
        concentrations = self.concentrate(*self.stain(images, True) if src is None else src)
        images = self.dilute(concentrations, self._target if dst is None else dst).reshape(images.shape)
        images = self._to(images)
        return images[..., [2, 1, 0]] if self._reverse else images

    def split(self, images):
        shape = images.shape
        images, stain = self.stain(images, True)
        concentrations = self.concentrate(images, stain).reshape(list(shape[:-1]) + [-1])
        return stain, concentrations

    def merge(self, stain, concentrations):
        return self._to(self.dilute(concentrations, stain))

    def stain(self, images, reuse=False):
        images = images[..., [2, 1, 0]] if self._reverse else images
        images = self._brightness(images)
        images, params = self.core(images)
        if reuse:
            return images, params
        else:
            return params

    def core(self, images):
        raise NotImplementedError

    def concentrate(self, *args, **kwargs):
        raise NotImplementedError

    def dilute(self, *args, **kwargs):
        raise NotImplementedError


class ReinhardNormal(Normal):
    def core(self, images):
        images = self._from(images)
        params = images.mean(axis=(0, 1)), images.std(axis=(0, 1))
        return images, params

    def concentrate(self, images, stain):
        return (images - stain[0]) / stain[1]

    def dilute(self, images, stain):
        images = (images * stain[1] + stain[0])
        return images.clip(0, 255).astype('uint8')


class SpamsNormal(Normal):
    def __init__(self, target=None, brightness=90, reverse=False):
        super(SpamsNormal, self).__init__(conversion.COLOR2OD(), conversion.OD2COLOR(), target, brightness, reverse)

    def concentrate(self, images, stain, lamda=0.01):
        images, stain = images.T, stain.T
        if not np.isfortran(images):
            images = np.asfortranarray(images)
        if not np.isfortran(stain):
            stain = np.asfortranarray(stain)
        return spams.lasso(images, D=stain, mode=2, lambda1=lamda, pos=True).toarray().T

    def dilute(self, images, stain):
        return np.dot(images, stain)

    @staticmethod
    def normalize(matrix):
        return matrix / np.linalg.norm(matrix, axis=1)[:, None]


class VahadaneNormal(SpamsNormal):
    def __init__(self, target=None, brightness=90, threshold=0.8, lamda=0.1, reverse=False):
        self._threshold = threshold
        self._lamda = lamda
        super(VahadaneNormal, self).__init__(target, brightness, reverse)

    def mask(self, image):
        shape = image.shape
        lab = cv2.cvtColor(image.reshape([-1, shape[-2], shape[-1]]), cv2.COLOR_RGB2LAB).reshape(shape)
        return lab[:, :, 0] / 255.0 < self._threshold

    def core(self, images):
        """
        Get 2x3 stain matrix. First row H and second row E
        :param image:
        :param threshold:
        :param lamda:
        :return:
        """
        mask = self.mask(images).reshape((-1,))
        images = self._from(images).reshape((-1, 3))
        dictionary = spams.trainDL(
            images[mask].T, K=2, lambda1=self._lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False
        ).T
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]
        return images, self.normalize(dictionary)


class MacenkoNormal(SpamsNormal):
    def __init__(self, target=None, brightness=90, threshold=0.15, ratio=99, reverse=False):
        self._threshold = threshold
        self._ratio = ratio
        super(MacenkoNormal, self).__init__(target, brightness, reverse)

    def core(self, image):
        image = self._from(image).reshape([-1, 3])
        optical_density = (image[(image > self._threshold).any(axis=1), :])
        _, v = np.linalg.eigh(np.cov(optical_density, rowvar=False))
        v = v[:, [2, 1]]
        if v[0, 0] < 0: v[:, 0] *= -1
        if v[0, 1] < 0: v[:, 1] *= -1
        that = np.dot(optical_density, v)
        phi = np.arctan2(that[:, 1], that[:, 0])
        min = np.percentile(phi, 100 - self._ratio)
        max = np.percentile(phi, self._ratio)
        v1 = np.dot(v, np.array([np.cos(min), np.sin(min)]))
        v2 = np.dot(v, np.array([np.cos(max), np.sin(max)]))
        if v1[0] > v2[0]:
            he = np.array([v1, v2])
        else:
            he = np.array([v2, v1])
        matrix = self.normalize(he)
        concentrations = super(MacenkoNormal, self).concentrate(image, matrix)
        const = np.percentile(concentrations, self._ratio, axis=0).reshape((1, 2))
        return image, (matrix, const)

    def concentrate(self, images, stain, lamda=0.01):
        return super(MacenkoNormal, self).concentrate(images, stain[0]) / stain[1]

    def dilute(self, images, stain):
        return super(MacenkoNormal, self).dilute(images * stain[1], stain[0])


class ReinhardNormalRGB(ReinhardNormal):
    def __init__(self, target=None, brightness=90):
        super(ReinhardNormal, self).__init__(
            lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB),
            lambda x: cv2.cvtColor(x, cv2.COLOR_LAB2RGB), target, brightness)


class ReinhardNormalBGR(ReinhardNormal):
    def __init__(self, target=None, brightness=90):
        super(ReinhardNormal, self).__init__(
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2LAB),
            lambda x: cv2.cvtColor(x, cv2.COLOR_LAB2BGR), target, brightness)


class VahadaneNormalRGB(VahadaneNormal):
    def __init__(self, target=None, brightness=90, threshold=0.8, lamda=0.1):
        super(VahadaneNormalRGB, self).__init__(target, brightness, threshold, lamda, reverse=False)


class VahadaneNormalBGR(VahadaneNormal):
    def __init__(self, target=None, brightness=90, threshold=0.8, lamda=0.1):
        super(VahadaneNormalBGR, self).__init__(target, brightness, threshold, lamda, reverse=True)


class MacenkoNormalRGB(MacenkoNormal):
    def __init__(self, target=None, brightness=90, threshold=0.15, ratio=99):
        super(MacenkoNormalRGB, self).__init__(target, brightness, threshold, ratio, reverse=False)


class MacenkoNormalBGR(MacenkoNormal):
    def __init__(self, target=None, brightness=90, threshold=0.15, ratio=99):
        super(MacenkoNormalBGR, self).__init__(target, brightness, threshold, ratio, reverse=True)
