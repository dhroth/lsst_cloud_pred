from __future__ import division
from __future__ import print_function

import numpy as np
import healpy as hp
import cartesianSky
from cartesianSky import CartesianSky

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
dirs += [10*W, 10*N, 10*E, 10*S]

#dirs = [W, N, E, S]

def predClouds(pastHpix, nowHpix, numSecs):
    """ Predict what the cloud map will be in numSecs secs

    First, convert pastHpix and nowPix to cartesian maps, since
    we model clouds as shapes moving in a flat 2d space.

    Then, translate pastCart in all possible directions and calculate
    the rmse between the translated map and nowCart. Choose the
    translated map with the lowest rmse. Continue translating
    until we reach a minimum rmse.

    This gives a direction vector which we can multiply by
    numSecs and apply to nowCart to generate the predicted cloud
    map, which is returned in hpix format

    @returns    healpix map representing the predicted cloud map
                in numSecs
    @param      pastHpix: a healpix of the cloud map 5 minutes ago
                (5 minutes is arbitrary and can easily be changed)
    @param      nowHpix: a healpix of the current cloud map
    @param      numSecs: the number of seconds into the future to
                make a prediction for
    @throws     ValueError if pastHpix and curHpix have different
                sizes or if numSecs is <= 0
    """

    if pastHpix.size != nowHpix.size:
        raise ValueError("pastHpix and nowHpix must have the same size")
    if numSecs <= 0:
        raise ValueError("numSecs must be >= 0")

    # convert the two healpix maps to cartesian maps
    pastCart = cartesianSky.fromHpix(pastHpix)
    #pastCart.plot(pastCart.max(), "past")
    #pastCart.translate(np.array([0,20])).plot(pastCart.max(), "trans")
    #plt.show()
    nowCart = cartesianSky.fromHpix(nowHpix)

    pool = Pool(len(dirs))
    
    # overallTrans is the translation necessary to translate
    # pastCart into nowCart (or at least get as close as possible)
    overallTrans = np.array([0,0])

    # this counter maps directions which are a local minimum to the 
    # number of times we've visited that minimum
    localMinimaNumVisits = Counter()

    while True:
        # get the rmse if we stop translating now
        stationaryRmse = calcRmse((nowCart, pastCart, overallTrans))
        
        # calculate the rmse between pastCart and nowCart when
        # the two have been offset from each other by many directions
        # around the current overallTrans
        testDirs = [overallTrans + direction for direction in dirs]
        args = [(nowCart, pastCart, testDir) for testDir in testDirs]
        rmses = pool.map(calcRmse, args)
        
        # figure out which direction yields the smallest rmse
        minDirId = np.argmin(rmses)
        minDir = testDirs[minDirId]
        minRmse = rmses[minDirId]

        # if translating nowCart in all directions yields an increase in
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
    # the translation needed to transform nowCart into predCart
    # also multiply by -1 because overallTrans is in the wrong direction
    scaleFactor = numSecs / (5 * 60.0)
    predTrans = -1 * np.round(overallTrans * scaleFactor).astype(int)
    predCart = nowCart.translate(predTrans)

    """ Print out the cartesian maps for debugging"""
    translatedPastCart = pastCart.translate(-1 * overallTrans)
    maxPix = max(pastCart.max(), nowCart.max(),
                 translatedPastCart.max(), predCart.max()) 
    pastCart.plot(maxPix, "pastCart")
    nowCart.plot(maxPix, "nowCart")
    translatedPastCart.plot(maxPix, "translatedPastCart")
    predCart.plot(maxPix, "predCart")
    """"""

    return cartesianSky.toHpix(predCart)

def calcRmse((cart1, cart2, direction)):
    """ Calculate the rmse between cart1 and cart2 when cart2 is shifted by dir

    @returns    the root mean squared error
    @param      cart1: the stationary map
    @param      cart2: the map which is shifted
    @param      direction: a list [y,x] specifying the direction to translate
    @throws     TypeError if cart1 and cart2 are invalid CartesianSky instances
    """

    if not isinstance(cart1, CartesianSky):
        raise TypeError("cart1 must be a CartesianSky object")
    if not isinstance(cart2, CartesianSky):
        raise TypeError("cart2 must be a CartesianSky object")

    # Only loop over the pixels which overlap after shifting cart2
    """
    Consider the following two scenarios. In each, the "o" marks
    (yStart, xStart) and the "x" marks (yEnd, xEnd). In the first
    case, both dimensions of direction are positive so the starting
    point is in the middle of cart1. In the second case, one dimension
    of direction is negative so the starting point is on an edge of
    cart1.
    
        ___________
        |         | cart1
        |   ___________
        |   |o    |   | cart2
        |   |    x|   |
        ----|------   |
            |         |
            -----------

        ___________
        |         |
    ___________   |
    |   |o    |   |
    |   |    x|   | cart1
    |   ------|----
    |         |  cart2
    ----------- 
    """

    xyMax = cartesianSky.xyMax

    yStart = max(0, direction[0])
    xStart = max(0, direction[1])
    yEnd = min(xyMax, xyMax + direction[0])
    xEnd = min(xyMax, xyMax + direction[1])

    ySign = np.sign(direction[0])
    xSign = np.sign(direction[1])

    mse = 0
    numPix = 0
    for y in range(yStart, yEnd, 2):
        for x in range(xStart, xEnd, 2):
            yOff = y - ySign * direction[0]
            xOff = x - xSign * direction[1]
            
            # ignore pixels which are not valid for both maps
            if not cart1.isPixelValid([y,x]):
                continue
            if not cart2.isPixelValid([yOff,xOff]):
                continue

            # TODO does this belong here?
            #if cart1[y,x] == -1 or cart2[yOff,xOff] == -1:
            #    raise ValueError("there must be no unseen valid pixels")

            mse += (cart1[y,x] - cart2[yOff,xOff])**2
            numPix += 1
    mse /= numPix
    return np.sqrt(mse)

