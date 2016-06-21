from __future__ import division
from __future__ import print_function

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from predClouds import predClouds
import cartesianSky
from cartesianSky import CartesianSky

from astropy.io import fits

def fits2Hpix(fits):
    """ Convert a fits image to a healpix map

    @returns    a healpix map with the fits data in it
    @param      a fits image array

    This function maps a fits allsky image into a healpix map, cutting
    the fits image off at a maximum zenith angle.
    """

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
    x = np.repeat([np.arange(-xCent, xCent)], yMax, axis=0).T
    y = np.repeat([np.arange(-yCent, yCent)], xMax, axis=0)
    
    # TODO probably don't need cart at all any more? Also where used
    # in hpix2Cartesian

    # cart[x,y] are cartesian coordinates in the focal plane
    # cart is yMax x xMax x 2, where cart[yindex,xindex] are
    # the pixel coordinates [y,x] corresponding to array indices
    # (yindex, xindex)
    cart = np.swapaxes([y,x],0,2)

    # (r,phi) are polar coordinates in the focal plane
    r = np.linalg.norm(cart, axis=2)

    # now calculating phi and theta is straightforward
    phi = np.arctan2(y, x).T

    # theta is the zenith angle
    theta = 2 * np.arcsin(r / (2 * f))

    # TODO are the 3 arrays in the fits file actually (r,g,b)? When I 
    # tried pylab.imshow() on the fits.data, the image looked rather red
    # I generated the fits files using the raw2fits script
    # the fits files that I got from Chris in the ut111515 directory only
    # have one subarray it seems so I'll just pretend it's blue for now
    #(r,g,b) = fits
    b = fits

    b -= bias

    b[theta > cartesianSky.thetaMax] = -1

    # the blue probably has the most information, so ignore r and g
    hpix = np.zeros(cartesianSky.npix)
    hpix[hp.ang2pix(cartesianSky.nside, theta, phi)] = b
    
    return hpix

if __name__ == "__main__":
    # these two are files I converted with raw2fits
    #filename1 = "fits/ut042816.daycal.0920.fits"
    #filename2 = "fits/ut042816.daycal.0980.fits"
    
    # these are files Chris put in /data/allsky/ut111515
    dataDir = "./"
    filePrefix = "ut111515.daycal."
    filePostfix = ".fits"
    def getFilename(filenum):
        return dataDir + filePrefix + filenum + filePostfix

    pastFilename   = getFilename("0250")
    nowFilename    = getFilename("0269")
    futureFilename = getFilename("0288")

    pastHpix   = fits2Hpix(fits.open(pastFilename  )[0].data)
    nowHpix    = fits2Hpix(fits.open(nowFilename   )[0].data)
    futureHpix = fits2Hpix(fits.open(futureFilename)[0].data)

    # convert the two healpix maps to cartesian maps
    pastCart   = cartesianSky.fromHpix(pastHpix)
    nowCart    = cartesianSky.fromHpix(nowHpix)
    futureCart = cartesianSky.fromHpix(futureHpix)

    # run the prediction 
    predCart = predClouds(pastCart, nowCart, 5 * 60)

    maxPix = max(pastCart.max(), nowCart.max(), 
                 predCart.max(), futureCart.max())

    pastCart.plot(maxPix, "past")
    nowCart.plot(maxPix, "now")
    predCart.plot(maxPix, "pred")
    futureCart.plot(maxPix, "future")

    # calculate various forms of accuracy

    # cloudyThreshold would presumably be determined by the tolerance
    # LSST has for looking through clouds
    cloudyThreshold = 1000 
    numPix = 0
    numPredCloudyPix = 0
    numFutureCloudyPix = 0
    numPredAndFutureCloudyPix = 0
    numPredClearAndFutureCloudyPix = 0
    numPredCloudyAndFutureClearPix = 0
    for y in range(cartesianSky.xyMax):
        for x in range(cartesianSky.xyMax):
            if (not predCart.isPixelValid([y,x]) or
                not futureCart.isPixelValid([y,x])):
                continue
            isPredCloudy = predCart[y,x] > cloudyThreshold
            isFutureCloudy = futureCart[y,x] > cloudyThreshold

            numPix += 1 
            if isPredCloudy:  
                numPredCloudyPix += 1
            if isFutureCloudy:
                numFutureCloudyPix += 1
            if isPredCloudy and isFutureCloudy:
                numPredAndFutureCloudyPix += 1
            if not isPredCloudy and isFutureCloudy:
                numPredClearAndFutureCloudyPix += 1
            if isPredCloudy and not isFutureCloudy:
                numPredCloudyAndFutureClearPix += 1
    print("Of the pixels which turned out to be cloudy, ",
          numPredAndFutureCloudyPix / numFutureCloudyPix * 100,
          "percent of them were predicted to be cloudy.")
    print("Of the pixels which turned out to be cloudy,",
          numPredClearAndFutureCloudyPix / numFutureCloudyPix * 100,
          "percent of them were predicted to be clear.")
    print("Of the pixels which turned out to be clear,",
          numPredCloudyAndFutureClearPix / (numPix - numFutureCloudyPix) * 100,
          "percent of them were predicted to be cloudy.")

    plt.show()
