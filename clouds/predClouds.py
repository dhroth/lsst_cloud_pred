from __future__ import division
from __future__ import print_function

import numpy as np
import cloudMap
from cloudMap import CloudMap
from cloudState import CloudState

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from multiprocessing import Pool
from collections import Counter

# there are 8 nearest pixel neighbors: SW, W, NW, N, NE, E, SE, S
SW = np.array([-1, -1])
W  = np.array([ 0, -1])
NW = np.array([ 1, -1])
N  = np.array([ 1,  0])
NE = np.array([ 1,  1])
E  = np.array([ 0,  1])
SE = np.array([-1,  1])
S  = np.array([-1,  0])

# each dir is considered in parallel so you may not want more
# dirs than you have cores
dirs = [SW, W, NW, N, NE, E, SE, S]
dirs += [2*W, 2*N, 2*E, 2*S]
dirs += [3*W, 3*N, 3*E, 3*S]
dirs += [4*W, 4*N, 4*E, 4*S]
#dirs += [10*W, 10*N, 10*E, 10*S]

#dirs = [W, N, E, S]

def predClouds(pastMap, nowMap, numSecs):
    """ Predict what the cloud map will be in numSecs secs

    Translate pastMap in all possible directions and calculate
    the rmse between the translated map and nowMap. Choose the
    translated map with the lowest rmse. Continue translating
    until we reach a minimum rmse.

    This gives a direction vector which we can multiply by
    numSecs and apply to nowMap to generate the predicted cloud
    map, which is returned in hpix format

    @returns    CloudMap representing the predicted cloud map in numSecs
    @param      pastMap: a CloudMap from 5 minutes ago
                (5 minutes is arbitrary and can easily be changed)
    @param      nowMap: a current CloudMap
    @param      numSecs: the number of seconds into the future for which 
                to make a prediction
    @throws     ValueError if numSecs is <= 0
    @throws     TypeError if pastMap or nowMap are not CloudMap objects
    """

    if numSecs <= 0:
        raise ValueError("numSecs must be >= 0")
    if not isinstance(pastMap, CloudMap):
        raise TypeError("pastMap must be a CloudMap instance")
    if not isinstance(nowMap, CloudMap):
        raise TypeError("nowMap must be a CloudMap instance")

    pool = Pool(len(dirs))
    
    # overallTrans is the translation necessary to translate
    # pastMap into nowMap (or at least get as close as possible)
    overallTrans = np.array([0,0])

    # this counter maps directions which are a local minimum to the 
    # number of times we've visited that minimum
    localMinimaNumVisits = Counter()

    while True:
        # get the rmse if we stop translating now
        stationaryRmse = calcRmse((nowMap, pastMap, overallTrans))
        
        # calculate the rmse between pastMap and nowMap when
        # the two have been offset from each other by many directions
        # around the current overallTrans
        testDirs = [overallTrans + direction for direction in dirs]
        args = [(nowMap, pastMap, testDir) for testDir in testDirs]
        rmses = pool.map(calcRmse, args)
        
        # figure out which direction yields the smallest rmse
        minDirId = np.argmin(rmses)
        minDir = testDirs[minDirId]
        minRmse = rmses[minDirId]

        # if translating nowMap in all directions yields an increase in
        # mse, then we've found a minimum of rmse. Keep track of local minima
        # so if we hit one enough times we can declare that we're done.

        # TODO it might improve the search to start at
        # some temperature and then gradually lower the temp over time
        if stationaryRmse <= minRmse:
            localMinimaNumVisits[tuple(overallTrans)] += 1
            # if we've revisited this minimum enough times then it's
            # probably a pretty stable minimum
            if localMinimaNumVisits[tuple(overallTrans)] > 1:
                break
        
        # now probabilistically choose a direction based on how close its
        # rmse is to the minimum rmse
        probas = [np.exp(minRmse - rmse) for rmse in rmses]
        probas /= np.sum(probas)
        chosenDirId = np.random.choice(len(testDirs), p=probas)
        chosenDir = testDirs[chosenDirId]
        print("translating in dir", chosenDir)
        overallTrans = chosenDir

    pool.close()
    pool.join()

    # now multiply overallTrans by numSecs / (5 minutes) to get
    # the translation needed to transform nowMap into predMap
    # also multiply by -1 because overallTrans is in the wrong direction
    scaleFactor = 1 / (5 * 60.0)
    predTrans = -1 * overallTrans * scaleFactor
    predMap = nowMap.transform(CloudState(predTrans), numSecs)

    """ Print out the cartesian maps for debugging
    translatedPastCart = pastCart.translate(-1 * overallTrans)
    maxPix = max(pastCart.max(), nowCart.max(),
                 translatedPastCart.max(), predCart.max()) 
    pastCart.plot(maxPix, "pastCart")
    nowCart.plot(maxPix, "nowCart")
    translatedPastCart.plot(maxPix, "translatedPastCart")
    predCart.plot(maxPix, "predCart")
    """

    return predMap

def calcRmse((map1, map2, direction)):
    """ Calculate the rmse between map1 and map2 when map2 is shifted by dir

    @returns    the root mean squared error
    @param      map1: the stationary map
    @param      map2: the map which is shifted
    @param      direction: a list [y,x] specifying the direction to translate
    @throws     TypeError if map1 and map2 are not CloudMap instances
    """

    if not isinstance(map1, CloudMap):
        raise TypeError("map1 must be a CloudMap object")
    if not isinstance(map2, CloudMap):
        raise TypeError("map2 must be a CloudMap object")

    # Only loop over the pixels which overlap after shifting map2
    """
    Consider the following two scenarios. In each, the "o" marks
    (yStart, xStart) and the "x" marks (yEnd, xEnd). In the first
    case, both dimensions of direction are positive so the starting
    point is in the middle of map1. In the second case, one dimension
    of direction is negative so the starting point is on an edge of
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
    for y in range(yStart, yEnd, 2):
        for x in range(xStart, xEnd, 2):
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

