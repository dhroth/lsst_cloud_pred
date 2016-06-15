from __future__ import division

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from predClouds import predClouds
# TODO kind of silly
from predClouds import nside

from astropy.io import fits

npix = hp.nside2npix(nside)

def fits2Hpix(fits):
    """ Convert a fits image to a healpix map

    @returns    a healpix map with the fits data in it
    @param      a fits image array

    This function maps a fits allsky image into a healpix map, cutting
    the fits image off at a maximum zenith angle.
    """

    # TODO this should probably be read from the fits files or something
    xMax = 2897
    yMax = 1935
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
    (r,g,b) = fits

    b -= bias

    # TODO parametrize
    b[theta > 70 * np.pi / 180] = 0

    # the blue probably has the most information, so ignore r and g
    nside = 32
    hpix = np.zeros(hp.nside2npix(nside))
    hpix[hp.ang2pix(nside, theta, phi)] = b
    
    return hpix

""" I put this in but haven't used or tested it. Not sure if we'll actually
need it since eventually this'll get run on night-time images
def removeSun(sky):
    # average the image to find the sun
    n = 5
    k = np.ones(n,n) / n**2
    avg = convolve2d(sky, k, mode="same")
    sunPos = np.unravel_index(avg.argmax(), avg.shape)
    
    isNearSun = np.linalg.norm(cart - sunPos) < sunStomp
    sky[isNearSun] = 0
"""

if __name__ == "__main__":
    filename1 = "fits/ut042816.daycal.0220.fits"
    filename2 = "fits/ut042816.daycal.0230.fits"

    past = fits2Hpix(fits.open(filename1)[0].data)
    now = fits2Hpix(fits.open(filename2)[0].data)

    maxPix = max(np.max(past),np.max(now))
    hp.mollview(past, max=maxPix, title="pastMollview")
    hp.mollview(now,  max=maxPix, title="nowMollview")
    
    # run the prediction 
    pred = predClouds(past, now, 5 * 60)

    hp.mollview(pred, max=maxPix)

    #plt.figure("predCart")
    #pylab.imshow(hpix2Cartesian(pred), vmax = 3000, cmap=plt.cm.jet)
    #plt.colorbar()

    plt.show()
