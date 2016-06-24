from __future__ import division
from __future__ import print_function

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from cloudServer import CloudServer
import cloudMap
from cloudMap import CloudMap

from astropy.io import fits
import os

# TODO this should probably be read from the fits files or something
#xMax = 2897
#yMax = 1935
xMax = 2888
yMax = 1924
xCent = xMax / 2
yCent = yMax / 2

# TODO 852 was in the matlab script but using that number made theta
# leave the allowed [0,pi] range, so I'm using rmax / 2 instead. Is
# this wrong?
f = 852 # effective focal length in pixels
f = np.sqrt(xCent**2 + yCent**2) / 2

# TODO could also read this in from the fits. Also may not be needed
bias = 2000 # approximately

# x and y are yMax x xMax, where, e.g., x[yindex, xindex]
# is the x coordinate corresponding to xindex (which is just
# xindex - xCenter)
y, x = np.ogrid[-yCent:yCent, -xCent:xCent]
#x = np.repeat([np.arange(-xCent, xCent)], yMax, axis=0).T
#y = np.repeat([np.arange(-yCent, yCent)], xMax, axis=0)

# TODO probably don't need cart at all any more? Also where used
# in hpix2Cartesian

# cart[x,y] are cartesian coordinates in the focal plane
# cart is yMax x xMax x 2, where cart[yindex,xindex] are
# the pixel coordinates [y,x] corresponding to array indices
# (yindex, xindex)
# cart = np.swapaxes([y,x],0,2)

# (r,phi) are polar coordinates in the focal plane
r = np.sqrt(y**2 + x**2)

# now calculating phi and theta is straightforward
phi = np.arctan2(y, x)

# theta is the zenith angle
theta = 2 * np.arcsin(r / (2 * f))

def fits2Hpix(fits):
    """ Convert a fits image to a healpix map

    @returns    a healpix map with the fits data in it
    @param      a fits image array

    This function maps a fits allsky image into a healpix map, cutting
    the fits image off at a maximum zenith angle.
    """


    # TODO are the 3 arrays in the fits file actually (r,g,b)? When I 
    # tried pylab.imshow() on the fits.data, the image looked rather red
    # I generated the fits files using the raw2fits script
    # the fits files that I got from Chris in the ut111515 directory only
    # have one subarray it seems so I'll just pretend it's blue for now
    #(r,g,b) = fits
    b = fits

    b -= bias

    b[theta > cloudMap.thetaMax] = -1

    # the blue probably has the most information, so ignore r and g
    hpix = np.zeros(cloudMap.npix)
    hpix[hp.ang2pix(cloudMap.nside, theta, phi)] = b
    
    return hpix

if __name__ == "__main__":
    dataDir = "/data/allsky/ut111515/"
    filePrefix = "ut111515.daycal."
    filePostfix = ".fits"
    def getFilename(filenum):
        return dataDir + filePrefix + str(filenum).zfill(4) + filePostfix

    # this is all of them
    nums = range(1,599)
    # this is the first ~1 hr chunk
    nums = range(0, 199)
    # this is the second ~1 hr chunk
    numStart = 201
    numEnd = 399
    numEnd = 240
    nums = range(numStart, numEnd)

    # first get mjds for each fits file
    mjds = np.zeros(len(nums)).astype(float)
    for i in nums:
        fileName = getFilename(i)
        if os.path.exists(fileName):
            mjd = fits.open(fileName)[0].header["mjd-obs"]
        else:
            mjd = -1
        mjds[i - numStart] = mjd

    # start up the cloud server
    cloudServer = CloudServer()

    # put placeholder zeros in each figure to specify the imshow settings
    # this is probably the wrong way of doing this but it doesn't 
    # particularly matter
    placeholder = np.zeros((cloudMap.xyMax, cloudMap.xyMax))

    # I think this sets the physical size of the output
    plt.figsize=(10,10)

    # create a 3x3 grid of subplots
    fig, axarr = pylab.subplots(3,3)
    imgs = np.array([[None for x in range(3)] for y in range(4)])

    # and put the placeholder image in all of them
    for y in range(3):
        for x in range(3):
            axarr[y,x].axis("off")
            imgs[y,x] = axarr[y,x].imshow(placeholder, vmax=3000, cmap=plt.cm.jet)

    # put titles on each subplot
    axarr[0,1].set_title("True Clouds")

    axarr[1,0].set_title("~5 Min Prediction")
    axarr[1,1].set_title("~10 Min Prediction")
    axarr[1,2].set_title("~20 Min Prediction")

    axarr[2,0].set_title("Diff from True")
    axarr[2,1].set_title("Diff from True")
    axarr[2,2].set_title("Diff from True")

    """
    axarr[3,0].set_title("Accuracy")
    axarr[3,1].set_title("Accuracy")
    axarr[3,2].set_title("Accuracy")
    """

    # get the first array index whose mjd is greater than the argument 
    def getClosestNum(mjd):
        for i in nums:
            if mjd > mjds[i - numStart] or mjds[i - numStart] == -1:
                continue
            return i
        return -1

    # keep track of all the predictions
    fiveMinPreds = [None] * len(nums)
    tenMinPreds = [None] * len(nums)
    twentyMinPreds = [None] * len(nums)

    # loop through ever image, predicting 5, 10, and 20 minutes ahead
    # at each step and storing the results in the arrays above
    for i in nums:
        fileName = getFilename(i)
        if not os.path.exists(fileName):
            # there are a couple missing files. Too bad
            continue

        fitsFile = fits.open(fileName)[0]

        # convert to a CloudMap
        hpix = fits2Hpix(fitsFile.data)
        cMap = cloudMap.fromHpix(str(i), hpix)

        # and post to the CloudServer
        cloudServer.postCloudMap(mjds[i - numStart], cMap)

        # TODO shouldn't expose this
        if i <= numStart + cloudServer._NUM_VEL_CALC_FRAMES:
            # can't pred before we have enough posted data
            continue

        # TODO clean up this i - numStart business. Also the duplicate code
        fiveMinNum = getClosestNum(mjds[i - numStart] + (5 * 60) / 24 / 3600)
        if fiveMinNum != -1:
            fiveMinPred = cloudServer.predCloudMap(mjds[fiveMinNum - numStart])
            fiveMinPreds[fiveMinNum - numStart] = fiveMinPred

        tenMinNum = getClosestNum(mjds[i - numStart] + (10 * 60) / 24 / 3600)
        if tenMinNum != -1:
            tenMinPred = cloudServer.predCloudMap(mjds[tenMinNum - numStart])
            tenMinPreds[tenMinNum - numStart] = tenMinPred

        twentyMinNum = getClosestNum(mjds[i - numStart] + (20 * 60) / 24 / 3600)
        if twentyMinNum != -1:
            twentyMinPred = cloudServer.predCloudMap(mjds[twentyMinNum - numStart])
            twentyMinPreds[twentyMinNum - numStart] = twentyMinPred

        # now plot all the images for number i
        # TODO should keep this all private--can probably override subtraction
        imgs[0,1].set_data(cMap.cloudData)
        if fiveMinPreds[i - numStart] is not None:
            imgs[1,0].set_data(fiveMinPreds[i - numStart].cloudData)
            diff = np.abs(fiveMinPreds[i - numStart].cloudData - cMap.cloudData)
            diff[np.logical_not(fiveMinPreds[i - numStart].validMask)] = 0
            diff[np.logical_not(cMap.validMask)] = 0
            imgs[2,0].set_data(diff)
        if tenMinPreds[i - numStart] is not None:
            imgs[1,1].set_data(tenMinPreds[i - numStart].cloudData)
            diff = np.abs(tenMinPreds[i - numStart].cloudData - cMap.cloudData)
            diff[np.logical_not(tenMinPreds[i - numStart].validMask)] = 0
            diff[np.logical_not(cMap.validMask)] = 0
            imgs[2,1].set_data(diff)
        if twentyMinPreds[i - numStart] is not None:
            imgs[1,2].set_data(twentyMinPreds[i - numStart].cloudData)
            diff = np.abs(twentyMinPreds[i - numStart].cloudData - cMap.cloudData)
            diff[np.logical_not(twentyMinPreds[i - numStart].validMask)] = 0
            diff[np.logical_not(cMap.validMask)] = 0
            imgs[2,2].set_data(diff)

        print("saving number", i)
        pylab.savefig("fullpngs/" + str(i) + ".png", dpi=200)


    """
    # calculate various forms of accuracy

    # cloudyThreshold would presumably be determined by the tolerance
    # LSST has for looking through clouds
    cloudyThreshold = 1000 
    numFutureCloudy = np.size(np.where(futureMap > cloudyThreshold)[0])
    numFutureClear  = np.size(np.where(futureMap < cloudyThreshold)[0])

    fracCloudyandCloudy = np.size(
        np.where((predMap > cloudyThreshold) & (futureMap > cloudyThreshold))[0]
    ) / numFutureCloudy

    print("Of the pixels which turned out to be cloudy, ",
          fracCloudyandCloudy * 100,
          "percent of them were predicted to be cloudy.")

    fracPredClearAndFutureCloudy = np.size(
        np.where((predMap < cloudyThreshold) & (futureMap > cloudyThreshold))[0]
    ) / numFutureCloudy

    print("Of the pixels which turned out to be cloudy,",
          fracPredClearAndFutureCloudy * 100,
          "percent of them were predicted to be clear.")

    fracPredCloudyAndFutureClear = np.size(
        np.where((predMap > cloudyThreshold) & (futureMap < cloudyThreshold))[0]
    ) / numFutureClear

    print("Of the pixels which turned out to be clear,",
          fracPredCloudyAndFutureClear * 100,
          "percent of them were predicted to be cloudy.")
    """
