


def CloudPredictor(object):

    def __init__(self, imageList=None, mjds=None, nImages=8):
        """
        """
        self.nImages = nImages

        if imageList is not None:
            self.imageList = []
            self.mjds = []
        else:
            self.imageList = imageList
            self.mjds = mjds
            self.vx = []
            self.vy = []
            # loop over the pairs and compute the vx and vy list


    def addImage(self, image, mjd):
        """
        Add a new image to the list
        """

        self.imageList.append(image)
        self.mjds.append(mjd)

        if len(self.imageList) > self.nImages:
            self.imageList.pop(0)
            self.mjds.pop(0)
            self.vx.pop(0)
            self.vy.pop(0)

        # XXX--or maybe only compute this part if predictCloudMap is called.  Then lots of images
        # can roll through, but if the scheduler doesn't need a cloud map it doesn't do the expensive calcs
        # Compute a new vx, vy using self.image[-1] and self.imageList[-2]
        vx,vy = self._fitVxVy(self.image[-1], self.imageList[-2])
        self.vx.append(vx)
        self.vy.append(vy)


    def _fitVxVy(self, image1,image2, mjd1, mjd2):
        """
        Find the best fitting shift between 2 all-sky images
        """

        # Run a check to see if the images look all-clear.


    def predictCloudMap(self, mjd):
        """
        
        """
