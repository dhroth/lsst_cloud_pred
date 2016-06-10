import numpy as np
import healpy as hp
import math

import matplotlib.pyplot as plt

nside = 8
npix = hp.nside2npix(nside)

# there are 8 possible directions to translate the healpix:
# SW, W, NW, N, NE, E, SE, S
ndirs = 8

# sorry for the neighbo(u)rs spelling...

# neighbors is ndirs x npix where the (i,j)th entry is
# the pixel number that pixel j ends up at when translated
# one unit in the ith direction
neighbors = hp.get_all_neighbours(nside, np.arange(npix))

# stationary is also ndirs x npix where the (i,j)th entry
# is equal to j (or the pixel number that j ends up at when
# not translated at all)
stationary = np.vstack([np.arange(npix)] * ndirs)

# if neighbors[i,j] == -1, that means that the jth pixel
# has no neighbor in the ith direction. So we just keep that
# pixel in the same place and hope that it doesn't affect the
# prediction too much.
immobilePixels = neighbors == -1
mobilePixels = neighbors != -1
neighbors = mobilePixels * neighbors + immobilePixels * stationary


def predClouds(pastHpix, nowHpix, numSecs):
    """ Predict what the cloud map will be in numSecs secs

    Translate pastHpix in all possible directions and calculate
    the mse between the translated map and nowHpix. Choose the
    translated map with the lowest mse. Continue translating
    until we reach a minimum mse.

    This gives a direction vector which we can multiply by
    numSecs and apply to nowHpix to generate the predicted cloud
    map.

    @returns    healpix map representing the predicted cloud map
                in numSecs
    @param      pastHpix: a healpix of the cloud map 5 minutes ago
                (5 minutes is arbitrary and can easily change)
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
    
    # transHistory is a list of all translations needed to transform
    # pastHpix into curHpix (or at least get as close as possible)
    transHistory = []

    # curHpix is the translated version of pastHpix at each step
    curHpix = pastHpix
    while True:
        # get the mse if we don't translate translatedHpix
        stationaryMse = calcMse(curHpix, nowHpix)
        
        # try translating curHpix in every direction, 
        # calculating the mse for each direction
        testHpixes = [translateHpix(curHpix, i) for i in range(ndirs)]
        mses = [calcMse(testHpixes[i], nowHpix) for i in range(ndirs)]

        # figure out which direction yields the smallest mse
        minDir = np.argmin(mses)
        minMse = mses[minDir]

        # if translating curHpix in any direction yields an increase in
        # mse, then we've found the (local) minimum of mse so we're done
        if stationaryMse <= minMse:
            break
        else:
            print "translating in dir", minDir
            transHistory.append(minDir)
            curHpix = testHpixes[minDir]

    # calculate a deltaTheta and deltaPhi for the overall translation
    # do this by figuring out where the origin pixel (chosen arbitrarily)
    # ends up after every translation in transHistory
    originPixStart = hp.ang2pix(nside, math.pi * 0.5, 0)
    originPixEnd = originPixStart
    for i in range(len(transHistory)):
        transDir = transHistory[i]
        originPixEnd = neighbors[transDir, originPixEnd]

    (startTheta, startPhi) = hp.pix2ang(nside, originPixStart)
    (endTheta, endPhi) = hp.pix2ang(nside, originPixEnd)

    deltaTheta = endTheta - startTheta
    deltaPhi = endPhi - startPhi

    # now multiply deltaTheta and deltaPhi by numSecs / (5 minutes) to get
    # the translation needed to transform nowHpix to the predicted hpix
    scaleFactor = numSecs / (5 * 60.0)

    (nowTheta, nowPhi) = hp.pix2ang(nside, np.arange(npix))

    predTheta = (nowTheta + deltaTheta * scaleFactor) % (math.pi)
    predPhi   = (nowPhi   + deltaPhi   * scaleFactor) % (2 * math.pi) 

    predTrans = hp.ang2pix(nside, predTheta, predPhi)
    predHpix = nowHpix[predTrans]

    return predHpix

def translateHpix(hpix, direction):
    neighborsInDirection = neighbors[direction]
    return hpix[neighborsInDirection]

def calcMse(hpix1, hpix2):
    if hpix1.size != hpix2.size:
        raise ValueError("hpix1 and hpix2 must be the same size")

    mse = 0
    for i in range(hpix1.size):
        mse += (hpix1[i] - hpix2[i])**2
    return mse

if __name__ == "__main__":
    past = np.zeros(hp.nside2npix(2))
    past[27] = 1
    past = hp.ud_grade(past, nside_out=nside)

    now = np.zeros(hp.nside2npix(2))
    now[20] = 1
    now = hp.ud_grade(now, nside_out=nside)

    pred = predClouds(past, now, 5 * 60)

    hp.mollview(past, min=0, max=1)
    hp.mollview(now,  min=0, max=1)
    hp.mollview(pred, min=0, max=1)

    plt.show()
