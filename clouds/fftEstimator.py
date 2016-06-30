from __future__ import division
from __future__ import print_function

import copy
import numpy as np
from skimage.feature import register_translation
from scipy.signal import fftconvolve
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from cloudStateEstimator import CloudStateEstimator
from cloudMap import CloudMap
import cloudMap
from cloudState import CloudState


class FftEstimator(CloudStateEstimator):
    @staticmethod
    def _doEstimateCloudState(map1, map2, deltaT):
        noiseAdded1 = FftEstimator._addNoise(map1).cloudData
        noiseAdded2 = FftEstimator._addNoise(map2).cloudData

        blurred1 = gaussian_filter(noiseAdded1, 1)
        blurred2 = gaussian_filter(noiseAdded2, 1)
        shift, _, _ = register_translation(blurred2, blurred1, 10)
        """
        corr = fftconvolve(map1.cloudData, map2.cloudData[::-1, ::-1])
        weight = fftconvolve(map1.validMask, map2.validMask[::-1, ::-1])
        corr = corr / weight
        peak = np.unravel_index(corr.argmax(), corr.shape)[::-1]
        shift = (peak[0] - cloudMap.xyCent, peak[1] - cloudMap.xyCent)
        """
        print("shift:", shift)
        cloudVelocity = shift / deltaT
        print(cloudVelocity)
        return CloudState(vel=cloudVelocity)
    
    @staticmethod
    def _addNoise(cloudMap):
        mapCopy = copy.deepcopy(cloudMap)
        invalidPix = np.where(np.logical_not(cloudMap.validMask))
        mean = mapCopy.mean()
        std = mapCopy.std()
        noise = np.random.normal(mean, std, size=invalidPix[0].shape)
        mapCopy.cloudData[invalidPix] = noise
        return mapCopy
