from __future__ import division
from __future__ import print_function

import numpy as np
import healpy as hp

from scipy.signal import convolve2d

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

nside = 32
npix = hp.nside2npix(nside)

# ignore pixels in healpix maps with theta > thetaMax
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

# minimum distance from the sun in pixels 
sunAvoidRadius = 30

class CartesianSky:
    """ Cartesian representation of the sky

    Instance variables:
    cart:       a numpy array of shape (xyMax, xyMax) mapping the cloud cover
    sunPos:     the position [y,x] of the sun in this image
    validMask:  a binary mask indicating which pixels in the image are valid

    Methods:
    __init__(cart, sunPos): constructor
    isPixelValid(point):    return whether point is part of the sky map
    getSunPos():            find the sun in this sky map
    translate(direction):   translate the map in some direction
    __getitem__((y,x)):     allows for syntax like cart[y,x]
    plot(maxPixel, title):  plot the map
    max():                  calculate the maximum pixel value
    std():                  calculate the std dev of the valid pixels
    mean():                 calculate the mean of the valid pixels
    """

    def __init__(self, cart, sunPos = None):
        """ Initialize the CartesianSky

        @returns    void
        @param      cart: a np.array with the cloud cover pixel values
        @param      sunPos (optional): the position of the sun in the image
                    If not passed in, it will be calculated
        @throws     ValueError if cart has the wrong shape or sunPos is
                    outside of the image

        Calculates the sun position if none is passed in and then calculates
        the valid mask for the sky map.

        """
        if cart.shape != (xyMax, xyMax):
            raise ValueError("the passed in map has the wrong shape")
        if sunPos is not None and (sunPos[0] < 0 or sunPos[1] < 0 or
                                   sunPos[0] > xyMax or sunPos[1] > xyMax):
            raise ValueError("the passed-in sunPos is invalid")

        self.cart = cart
        # allow the caller to pass in a sunPos if it's already known
        if sunPos is None:
            self.sunPos = self.getSunPos()
        else:
            self.sunPos = sunPos
        # keep track of which pixels are valid
        self.validMask = np.array([[self.isPixelValid([y,x]) 
                                   for x in range(xyMax)] 
                                   for y in range(xyMax)])
        # mark invalid pixels as -1
        self.cart[np.logical_not(self.validMask)] = -1

    def isPixelValid(self, point):
        """ Return whether the pixel is a valid part of the sky map

        @returns    True if the pixel is valid, False otherwise
        @param      point: the pixel in question
        @throws     ValueError if point is outside the image

        A pixel is invalid if it's outside of rMax or too close to the sun.
        """
        if point[0] < 0 or point[1] < 0 or point[0] > xyMax or point[1] > xyMax:
            return False
            # raise ValueError("the supplied point is outside the sky map")

        point = np.array(point)
        center = np.array([xyCent,xyCent])

        # return False if the pixel is farther than rMax from the center
        # or if it is closer than sunAvoidRadius from the sun
        if np.linalg.norm(point - center) > rMax:
            return False
        if np.linalg.norm(point - self.sunPos) < sunAvoidRadius:
            return False

        return True

    def __getitem__(self, args):
        # given a CartesianSky object cart, this method allows other
        # code to use cart[y,x] instead of having to do cart.cart[y,x]
        # which would breach abstraction anyway
        (y, x) = args
        return self.cart[y,x]

    def getSunPos(self):
        """ Find the position of the sun in the image
    
        @returns    a point [y,x] indicating the sun's position in self.cart
        
        This method smooths out the sky map and then finds the maximum
        pixel value in the smoothed image. If the smoothed image has
        multiple pixels which share the same maximum, this chooses
        the first such pixel.
        """
        # average the image to find the sun
        n = 5
        k = np.ones((n,n)) / n**2
        avg = convolve2d(self.cart, k, mode="same")

        sunPos = np.unravel_index(avg.argmax(), avg.shape)
        return sunPos

    def translate(self, direction):
        """ Translate our cartesian map in the specified direction

        Pixels which are translated from invalid points to valid points take
        the value of the old pixel.

        @returns    the translated map
        @param      direction: the direction [y,x] to translate by
        """

        # translate the array by padding it with zeros and then cropping off the
        # extra numbers
        if direction[0] >= 0:
            padY = (direction[0], 0)
        else:
            padY = (0, -1 * direction[0])
        if direction[1] >= 0:
            padX = (direction[1], 0)
        else:
            padX = (0, -1 * direction[1])

        paddedCart = np.pad(self.cart, (padY, padX), 
                            mode="constant", constant_values=-1)

        # now crop the image to the original size
        cropY = (padY[1], paddedCart.shape[0] - padY[0])
        cropX = (padX[1], paddedCart.shape[1] - padX[0])
        translatedCart = paddedCart[cropY[0]:cropY[1],cropX[0]:cropX[1]]

        # replace all pixels which are -1 with the value they used to be
        for y in range(xyMax):
            for x in range(xyMax):
                if translatedCart[y,x] == -1:
                    oldY = y - direction[0]
                    oldX = x - direction[1]
                    translatedCart[y,x] = self.cart[y,x]

        # np.roll translates with wrap around but we probably don't want this
        #translatedCart = np.roll(cart, direction[0], axis=0)
        #translatedCart = np.roll(translatedCart, direction[1], axis=1)

        return CartesianSky(translatedCart, sunPos = self.sunPos + direction)

    def plot(self, maxPixel, title=""):
        plt.figure(title)
        pylab.imshow(self.cart, vmax = maxPixel, cmap=plt.cm.jet)
        plt.colorbar()
        # uncomment plt.show() to pause after each call to this function
        #plt.show()

    def max(self):
        return np.max(self.cart)

    def std(self):
        return np.std(self.cart[self.validMask])

    def mean(self):
        return np.mean(self.cart[self.validMask])


def fromHpix(hpix):
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
    sky[sky == 0] = -1

    maxPix = np.max(sky)

    return CartesianSky(sky)

def toHpix(cart):
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
