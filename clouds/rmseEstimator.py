from __future__ import division

from cloudStateEstimator import CloudStateEstimator
from cloudState import CloudState
from cloudMap import CloudMap
import cloudMap
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline

import numpy as np


class RmseEstimator(CloudStateEstimator):

    # TODO this cache never gets cleared, which is a wide-open memory leak
    # this needs to be fixed if RmseEstimator is actually going to be used
    _cachedRmses = {}

    @staticmethod
    def _doEstimateCloudState(map1, map2, deltaT):
        # TODO
        # the parameters are vy and vx
        initialGuess = np.array([5,5])
        result = minimize(RmseEstimator._calcInterpolatedRmse, 
                          initialGuess, 
                          method="CG",
                          options={"eps":1},
                          args=(map1, map2))
        # TODO check result.success
        # calcRmse moves map2 around to make it look like map1. Since map2 is
        # from a time later than map1, that means that if, for example, you have
        # to move map2 down to make it look like map1, then the clouds moved up
        # between map1 and map2. Therefore we need the minus sign here
        cloudVelocity = -1 * result.x / deltaT
        #print("result of minimize:", cloudVelocity)
        return CloudState(vel=cloudVelocity)

    @staticmethod
    def _calcInterpolatedRmse(velocity, map1, map2):
        #print("interpolated:", velocity)
        (vy1, vx1) = np.floor(velocity).astype(int)
        (vy2, vx2) = np.ceil(velocity).astype(int)

        directions = [[vy1,vx1],[vy1,vx2],[vy2,vx1],[vy2,vx2]]
        rmses = [RmseEstimator._calcRmse(direction, map1, map2) 
                 for direction in directions]
        x = y = np.array([0,1])
        z = np.array(rmses).reshape(2,2)
        interpolator = RectBivariateSpline(x,y,z,kx=1,ky=1)
        return interpolator(velocity[0], velocity[1])[0][0]


    @staticmethod
    def _calcRmse(direction, map1, map2):
        """ Calculate rmse btwn map1 and map2 when map2 is shifted by velocity

        @returns    the root mean squared error
        TODO change once spread is added
        @param      velocity: the amount to shift map2 by
        @param      map1: the stationary map
        @param      map2: the map which is shifted
        @throws     TypeError if map1 and map2 are not CloudMap instances
        """

        if not isinstance(map1, CloudMap):
            raise TypeError("map1 must be a CloudMap object")
        if not isinstance(map2, CloudMap):
            raise TypeError("map2 must be a CloudMap object")

        # TODO the velocities passed in by scipy.optimize.minimize are floats
        # and I don't think one can easily change it to search over the integers
        # Perhaps a better way to do this would be to give scipy.optimize
        # a jacobian function instead of this? Would have to think about it 
        # some more
        #velocity = np.array(velocity) # just in case
        #direction = np.round(velocity).astype(int)

        # check if we've already calculated this rmse:
        calculationHash = (hash(tuple(direction)), map1.hash(), map2.hash())
        if calculationHash in RmseEstimator._cachedRmses:
            return RmseEstimator._cachedRmses[calculationHash]

        # Only loop over the pixels which overlap after shifting map2
        """
        Consider the following two scenarios. In each, the "o" marks
        (yStart, xStart) and the "x" marks (yEnd, xEnd). In the first
        case, both dimensions of velocity are positive so the starting
        point is in the middle of map1. In the second case, one dimension
        of velocity is negative so the starting point is on an edge of
        map1.
        
            ___________
            |         | map1
            |   ___________
            |   |o    |   | map2
            |   |    x|   |
            ----|------   |
                |         |
                -----------

            ___________
            |         |
        ___________   |
        |   |o    |   |
        |   |    x|   | map1
        |   ------|----
        |         |  map2
        ----------- 
        """

        xyMax = cloudMap.xyMax

        yStart = max(0, direction[0])
        xStart = max(0, direction[1])
        yEnd = min(xyMax, xyMax + direction[0])
        xEnd = min(xyMax, xyMax + direction[1])

        mse = 0
        numPix = 0
        for y in range(yStart, yEnd, 1):
            for x in range(xStart, xEnd, 1):
                # ignore pixels which are not valid for both maps
                if not map1.isPixelValid([y,x]):
                    continue
                yOff = y - direction[0]
                xOff = x - direction[1]

                if not map2.isPixelValid([yOff,xOff]):
                    continue

                # TODO does this belong here?
                #if map1[y,x] == -1 or map2[yOff,xOff] == -1:
                #    raise ValueError("there must be no unseen valid pixels")

                mse += (map1[y,x] - map2[yOff,xOff])**2
                numPix += 1
        mse /= numPix
        rmse = np.sqrt(mse)

        # cache so we don't recompute
        RmseEstimator._cachedRmses[calculationHash] = rmse
        return rmse


