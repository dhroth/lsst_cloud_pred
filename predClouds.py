from __future__ import division
from __future__ import print_function

import numpy as np
import healpy as hp
import math

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.signal import convolve2d

from multiprocessing import Pool
from collections import Counter

nside = 32
npix = hp.nside2npix(nside)

thetaMax = 70 * np.pi / 180

# use an XY plane with approximately somewhat more pixels than there 
# are in the hpix so we don't lose all resolution  at high theta
xyMax = int(np.sqrt(npix)) * 2
xyCent = int(xyMax / 2)

# when we convert hpix to cartesian, the resulting xy map is square
# but the signal is only found in a circle inscribed in the square
# (since the passed-in healpixes don't go all the way to the horizon)
# rMax is the radius in pixels of that circle
rMax = 0.9 * xyMax / 2

# z is the distance in pixels from the observer to the sky
# it is chosen to make the skymap fill our XY coordinates
# TODO I'm not sure I'm doing this right but it seems to work?
z = 40

# there are 8 nearest pixel neighbors: SW, W, NW, N, NE, E, SE, S
SW = np.array([-1, -1])
W  = np.array([ 0, -1])
NW = np.array([ 1, -1])
N  = np.array([ 1,  0])
NE = np.array([ 1,  1])
E  = np.array([ 0,  1])
SE = np.array([-1,  1])
S  = np.array([-1,  0])

# each dir is considered in parallel so you man not want more
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
    pastCart = hpix2Cartesian(pastHpix)
    nowCart = hpix2Cartesian(nowHpix)

    """ Print out the cartesian maps for debugging
    maxPix = max(np.max(pastCart), np.max(nowCart))
    maxPix = 3000
    fig1 = plt.figure("pastCart")
    pylab.imshow(pastCart, vmax = maxPix, cmap=plt.cm.jet)
    plt.colorbar()
    fig2 = plt.figure("nowCart")
    pylab.imshow(nowCart, vmax = maxPix, cmap=plt.cm.jet)
    plt.colorbar()
    """

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
        #rmses = [calcRmse(nowCart, pastCart, testDir) for testDir in testDirs]
        
        # figure out which direction yields the smallest rmse
        minDirId = np.argmin(rmses)
        minDir = testDirs[minDirId]
        minRmse = rmses[minDirId]

        # if translating nowCart in all directions yields an increase in
        # mse, then we've found the (local) minimum of mse so we're done
        # TODO it might improve the search to start at
        # some temperature and then gradually lower the temp over time
        if stationaryRmse <= minRmse:
            localMinimaNumVisits[tuple(overallTrans)] += 1
            # if we've revisited this minimum enough times then it's
            # probably a pretty stable minimum
            if localMinimaNumVisits[tuple(overallTrans)] > 5:
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
    predCart = translateCart(nowCart, predTrans)

    """ Print out predCart for debugging
    fig3 = plt.figure("predCartInPred")
    pylab.imshow(predCart, vmax = 3000, cmap = plt.cm.jet)
    plt.colorbar()
    """

    return cartesian2Hpix(predCart)

def translateCart(cart, direction):
    """ Translate the passed-in cartesian map in the specified direction

    The translation wraps around pixels which are bumped off the edge
    TODO not sure if that's what we want?

    @returns    the translated map
    @param      cart: the input map
    @param      direction: a list [y,x] of the direction to translate cart by
    """
    # translate the array the necessary amount in the y direction
    # and then the necessary amount in the x direction
    translatedCart = np.roll(cart, direction[0], axis=0)
    translatedCart = np.roll(translatedCart, direction[1], axis=1)

    return translatedCart

def calcRmse((cart1, cart2, direction)):
    """ Calculate the rmse between cart1 and cart2 when cart2 is shifted by dir

    @returns    the root mean squared error
    @param      cart1: the stationary map
    @param      cart2: the map which is shifted
    @param      direction: a list [y,x] specifying the direction to translate
    @throws     ValueError if cart1 and cart2 are different shapes or if
                they are not squares or if there are unseen pixels (=-1) inside
                of rMax
    """
    if cart1.shape != cart2.shape:
        raise ValueError("cart1 and cart2 must have the same shape")

    (yMax, xMax) = cart1.shape

    # We assume that the maps are squares with the signal in inscribed
    # circles. This conforms with the output of hpix2Cartesian()
    if yMax != xMax:
        raise ValueError("the passed in maps must be square")

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
    yStart = max(0, direction[0])
    xStart = max(0, direction[1])
    yEnd = min(yMax, yMax + direction[0])
    xEnd = min(xMax, xMax + direction[1])

    ySign = np.sign(direction[0])
    xSign = np.sign(direction[1])

    mse = 0
    numPix = 0

    #print(direction)
    for y in range(yStart, yEnd, 2):
        #print("\n", end="")
        for x in range(xStart, xEnd, 2):
            yOff = y - ySign * direction[0]
            xOff = x - xSign * direction[1]

            # don't count this pixel if it's outside rMax on either the
            # stationary or the translated maps
            if np.sqrt((y - xyCent)**2 + (x - xyCent)**2) > rMax:
                #print("x", end="")
                continue
            if np.sqrt((yOff - xyCent)**2 + (xOff - xyCent)**2) > rMax:
                #print("X", end="")
                continue
            #print("o", end="")

            if cart1[y,x] == -1 or cart2[yOff,xOff] == -1:
                raise ValueError("there must be no unseen pixels within rMax")

            mse += (cart1[y,x] - cart2[yOff,xOff])**2
            numPix += 1
    #print("\n")
    mse /= numPix
    return np.sqrt(mse)

def hpix2Cartesian(hpix):
    """ Convert a healpix image to a cartesian cloud map

    @returns    a square cartesian map with the signal in an inscribed circle
    @param      hpix: the healpix to be converted

    The top plane in the crude picture below is the cartesian plane
    where the clouds live. The dome is the healpix that we're 
    looking "through" to see the clouds
    _________________
         /   \
        |  o  |

    To find out which healpix pixel corresponds to (x,y), we convert 
    (x,y) to (r,phi). Then, we figure out which theta corresponds to
    the calculated r.
    """

    # now for each (x,y), sample the corresponding hpix pixel
    # see fits2Hpix() for an explanation of x, y, and cart
    x = np.repeat([np.arange(-xyCent, xyCent)], xyMax, axis=0).T
    y = np.repeat([np.arange(-xyCent, xyCent)], xyMax, axis=0)
    cart = np.swapaxes([y,x],0,2)
    
    # calculate theta and phi of each pixel in the cartesian map
    r = np.linalg.norm(cart, axis=2) 
    phi = np.arctan2(y, x).T
    theta = np.arctan(r / z)

    # ipixes is an array of pixel indices corresponding to theta and phi
    ipixes = hp.ang2pix(nside, theta, phi)

    # move back from physical coordinates to array indices
    y += xyCent
    x += xyCent
    y = y.astype(int) 
    x = x.astype(int)

    # set the sky pixels to the corresponding hpix pixel
    sky = np.zeros((xyMax, xyMax))
    sky[y.flatten(),x.flatten()] = hpix[ipixes.flatten()]
    sky[sky == 0] = np.median(sky)

    return sky

def cartesian2Hpix(cart):
    """ Convert a cartesian cloud map to a healpix image

    @returns    a healpix image of the clouds
    @param      cart: a cartesian map of the clouds
    
    For each pixel in hpix, sample from the corresponding pixel in cart
    """

    hpix = np.zeros(npix)
    (theta, phi) = hp.pix2ang(nside, np.arange(npix))

    r = np.tan(theta) * z
    x = np.floor(r * np.cos(phi)).astype(int)
    y = np.floor(r * np.sin(phi)).astype(int)
    
    # ignore all pixels with zenith angle higher than thetaMax
    x = x[theta < thetaMax]
    y = y[theta < thetaMax]
    ipixes = np.arange(npix)[theta < thetaMax]

    hpix[ipixes] = cart[x + xyCent, y + xyCent]

    return hpix
