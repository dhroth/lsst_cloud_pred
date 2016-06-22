from __future__ import division
from __future__ import print_function

import numpy as np
import cloudMap
from cloudMap import CloudMap
from cloudState import CloudState

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.optimize import minimize

class CloudServer:

    def __init__(self, maxCachedMaps=8):
        self.maxCachedMaps = maxCachedMaps
        self.cachedMaps = []

    def _calculateCloudState(self, map1, map2, deltaT):
        """ Find the cloud state using two closely-spaced CloudMap objects

        TODO update once spread is added
        Use scipy.optimize.minimize to find the velocity vector which minimizes
        the rmse between map1 and map2 when translated by the velocity.

        @returns    A best guess of the current dynamical CloudState
        @param      map1: a CloudMap of the clouds at some time
        @param      map2: a CloudMap of the clouds at a time deltaT after map1
        @param      deltaT: the number of seconds separating map1 and map2
        @throws     ValueError if deltaT < 0
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
        initialGuess = np.array([20,20])
        result = minimize(self._calcRmse, initialGuess, args=(map1, map2))
        # TODO check result.success
        cloudVelocity = result.x / deltaT

        return CloudState(vel=cloudVelocity)

    def postCloudMap(self, mjd, cloudMap):
        if len(self.cachedMaps) > 0:
            if mjd <= self.cachedMaps[-1].mjd:
                raise ValueError("cloud maps must be posted in order of mjd")
        self.cachedMaps.append(CachedMap(mjd, cloudMap))
        if len(self.cachedMaps) > self.maxCachedMaps:
            self.cachedMaps.pop(0)

    def predCloudMap(self, mjd):
        numMaps = len(self.cachedMaps)
        if numMaps == 0:
            raise RuntimeError("can't make predictions with no clouds posted")

        latestMap = self.cachedMaps[-1].cloudMap
        latestMjd = self.cachedMaps[-1].mjd
        if mjd <= latestMjd:
            raise ValueError("can't predict the past")

        # calculate cloudState for all pairs
        for i in range(1, numMaps):
            if self.cachedMaps[i].cloudState is None:
                map1 = self.cachedMaps[i-1].cloudMap
                map2 = self.cachedMaps[i].cloudMap
                deltaT = self.cachedMaps[i].mjd - self.cachedMaps[i-1].mjd
                self.cachedMaps[i].cloudState = \
                        self._calculateCloudState(map1, map2, deltaT)

        vys = [cachedMap.cloudState.vel[0] for cachedMap in self.cachedMaps[1:]]
        vxs = [cachedMap.cloudState.vel[1] for cachedMap in self.cachedMaps[1:]]
        
        v = [np.median(vys), np.median(vxs)]
        print("found velocity ", v)
        predMap = latestMap.transform(CloudState(vel=v), mjd - latestMjd)
        return predMap

    def _calcRmse(self, velocity, map1, map2):
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
        velocity = np.array(velocity) # just in case
        direction = np.round(velocity).astype(int)

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
        for y in range(yStart, yEnd, 3):
            for x in range(xStart, xEnd, 3):
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
        return np.sqrt(mse)

class CachedMap:
    def __init__(self, mjd, cloudMap, cloudState = None):
        self.mjd = mjd
        self.cloudMap = cloudMap
        self.cloudState = cloudState
