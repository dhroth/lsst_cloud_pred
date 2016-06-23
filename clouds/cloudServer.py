from __future__ import division
from __future__ import print_function

import numpy as np
import cloudMap
from cloudMap import CloudMap
from cloudState import CloudState

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline

import time

class CloudServer:

    def __init__(self):
        # throw out stale cloud maps once we reach more than this many
        self._MAX_CACHED_MAPS = 20
        # calculate velocity vectors between frames this far apart
        self._NUM_VEL_CALC_FRAMES = 10

        self._cachedMaps = []
        self._cachedRmses = {}

    def _calculateCloudState(self, map1, map2, deltaT):
        """ Find the cloud state using two closely-spaced CloudMap objects

        TODO update once spread is added
        Use scipy.optimize.minimize to find the velocity vector which minimizes
        the rmse between map1 and map2 when translated by the velocity.

        @returns    A best guess of the current dynamical CloudState
        @param      map1: a CloudMap of the clouds at some time
        @param      map2: a CloudMap of the clouds at a time deltaT after map1
        @param      deltaT: the difference in mjd between map1 and map2
        @throws     ValueError if deltaT <= 0
        @throws     TypeError if map1 or map2 are not CloudMap objects
        """

        if deltaT <= 0:
            raise ValueError("deltaT must be >= 0")
        if not isinstance(map1, CloudMap):
            raise TypeError("map1 must be a CloudMap instance")
        if not isinstance(map2, CloudMap):
            raise TypeError("map2 must be a CloudMap instance")

        # TODO
        # the parameters are vy and vx
        initialGuess = np.array([5,5])
        result = minimize(self._calcInterpolatedRmse, 
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
        print("result of minimize:", cloudVelocity)
        return CloudState(vel=cloudVelocity)

    def postCloudMap(self, mjd, cloudMap):
        """ Notify CloudServer that a new cloud map is available

        @returns    void
        @param      mjd: the time the image was taken
        @param      cloudMap: the cloud cover map
        @throws     ValueError if the caller attempts to post cloud maps
                    out of chronological order
        """

        if len(self._cachedMaps) > 0:
            if mjd <= self._cachedMaps[-1].mjd:
                raise ValueError("cloud maps must be posted in order of mjd")

        self._cachedMaps.append(CachedMap(mjd, cloudMap))
        if len(self._cachedMaps) > self._MAX_CACHED_MAPS:
            self._cachedMaps.pop(0)

    def predCloudMap(self, mjd):
        """ Predict the cloud map
        
        @returns    a CloudMap instance with the predicted cloud cover
        @param      mjd: the time the prediction is requested for
        @throws     RuntimeWarning if not enough cloud maps have been posted
        @throws     ValueError if mjd is before the latest posted cloud map
        """

        numMaps = len(self._cachedMaps)
        if numMaps <= self._NUM_VEL_CALC_FRAMES:
            raise RuntimeWarning("too few clouds have been posted to predict")

        latestMap = self._cachedMaps[-1].cloudMap
        latestMjd = self._cachedMaps[-1].mjd
        if mjd <= latestMjd:
            raise ValueError("can't predict the past")

        # calculate cloudState for all pairs
        for i in range(self._NUM_VEL_CALC_FRAMES, numMaps):
            if self._cachedMaps[i].cloudState is None:
                cachedMap1 = self._cachedMaps[i - self._NUM_VEL_CALC_FRAMES]
                cachedMap2 = self._cachedMaps[i]
                deltaT = cachedMap2.mjd - cachedMap1.mjd
                self._cachedMaps[i].cloudState = self._calculateCloudState(
                        cachedMap1.cloudMap, cachedMap2.cloudMap, deltaT
                     )

        vs = [cachedMap.cloudState.vel
                for cachedMap in self._cachedMaps[self._NUM_VEL_CALC_FRAMES:]]
        v = np.median(vs, axis=0)
        print("pred vs:", vs)
        print("pred median:", v)

        predMap = latestMap.transform(CloudState(vel=v), mjd - latestMjd)
        return predMap

    def _calcInterpolatedRmse(self, velocity, map1, map2):
        #print("interpolated:", velocity)
        (vy1, vx1) = np.floor(velocity).astype(int)
        (vy2, vx2) = np.ceil(velocity).astype(int)

        directions = [[vy1,vx1],[vy1,vx2],[vy2,vx1],[vy2,vx2]]
        rmses = [self._calcRmse(direction, map1, map2) for direction in directions]
        x = y = np.array([0,1])
        z = np.array(rmses).reshape(2,2)
        interpolator = RectBivariateSpline(x,y,z,kx=1,ky=1)
        return interpolator(velocity[0], velocity[1])[0][0]


    def _calcRmse(self, direction, map1, map2):
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
        map1Hash = map1.hash()
        map2Hash = map2.hash()
        directionHash = hash(tuple(direction))
        if (directionHash, map1Hash, map2Hash) in self._cachedRmses:
            return self._cachedRmses[(directionHash, map1Hash, map2Hash)]

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
        for y in range(yStart, yEnd, 4):
            for x in range(xStart, xEnd, 4):
                yOff = y - direction[0]
                xOff = x - direction[1]
                
                # ignore pixels which are not valid for both maps
                if not map1.isPixelValid([y,x]):
                    continue
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
        self._cachedRmses[(directionHash, map1Hash, map2Hash)] = rmse
        return rmse

class CachedMap:
    """ Wrapper class for parameters describing the clouds' dynamical state """
    def __init__(self, mjd, cloudMap, cloudState = None):
        self.mjd = mjd
        self.cloudMap = cloudMap
        self.cloudState = cloudState
