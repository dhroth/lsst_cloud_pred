from __future__ import division
from __future__ import print_function

from cloudStateEstimator import CloudStateEstimator
from cloudMap import CloudMap
from cloudState import CloudState

from skimage.feature import register_translation

class FftEstimator(CloudStateEstimator):
    @staticmethod
    def _doEstimateCloudState(map1, map2, deltaT):
        shift, _, _ = register_translation(map1.cloudData, map2.cloudData)
        return CloudState(vel=shift / deltaT)
