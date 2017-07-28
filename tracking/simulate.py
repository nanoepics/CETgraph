"""
    CETgraph.tracking.simulate.py
    ==================================
    Contains verious classes for generating synthetic images corresponding to particles performing a thermal Brownian
    motion to be viewed, and eventually analyzed when necessary.
    These classes can be used for example to test the reliability of the tracking algorithms

    .. lastedit:: 27/7/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np

class Waterfall:
    """Generates z-position vs time for a group of particles as they go with
    normal Brownian motion and drift.

    Parameters
    ----------
    size: 1D field of view in pixels
    n: number of particles in the field of view,
    difcon: diffusion constant (identical particles) [pixel^2/frame]

    size: field of view in pixels
    psize: particle size in pixels
    signal: brightness of each particle
    noise: background random noise
    drift: average drift velosity [pixel/frame]
    Returns
    -------
    float numpy array of intensity(position,time)
    """

    def __init__(self, fov = 500, n = 4, difcon = 1, signal = 10, noise = 1, psize = 8, drift = 1):
        self.fov = fov
        self.difcon = difcon
        self.drift = drift
        self.numpar = n
        self.signal = signal
        self.noise = noise
        self.psize = psize
        self.nframes = 100 # number of lines (frames) to be generated

    def genwf(self):
        positions = self.fov * np.random.rand(self.numpar)
        wf = np.zeros((self.fov, self.nframes))
        taxis = np.arange(self.nframes)
        for p in positions:  # generating random-walk assuming dt=1
            steps = np.random.standard_normal(self.nframes)
            path = p + np.cumsum(steps) * np.sqrt(self.difcon) + self.drift * taxis
            path[path > self.fov] -= self.fov
            wf[[np.asarray(path, dtype=int), taxis]] += self.signal
        fft_tracks = np.fft.rfft2(wf, axes=(-2,))
        max_freq = int(self.fov / self.psize)
        fft_tracks[max_freq:, :] = 0
        wf = abs(np.fft.irfft2(fft_tracks, axes=(-2,)))
        noise = np.random.randn(self.fov, self.nframes)
        wf += noise
        return wf


class SingleFrame:
    """
    :param size: [width, height] of the desired image that contains these particles
    :return: SingleFrame.loca: intended location of the particles (with sub-pixel resolution)
             SingleFrame.genImage: an image with specified noise and particles displaced accordingly
    """
    def __init__(self, fov = [300, 200], n = 4, difcon = 1, signal = 10, noise = 1, psize = 8):
        # camera and monitor parameters
        self.xfov, self.yfov = fov
        # simulation parameters
        self.difcon = difcon # Desired diffusion constant in pixel squared per frame
        self.numpar = n # Desired number of diffusing particles
        self.signal = signal # brightness for each particle
        self.noise = noise # background noise
        self.psize = psize # half-spread of each particle in the image, currently must be integer
        self.bg = self.genBG()
        self.initLocations()

    def initLocations(self):
        # initializes the random location of numpar particles in the frame. one can add more paramaters like intensity
        # and PSF distribution if necessary
        parx = np.random.uniform(0, self.xfov, size=(self.numpar, 1))
        pary = np.random.uniform(0, self.yfov, size=(self.numpar, 1))
        pari = np.random.uniform(1, self.numpar, size=(self.numpar, 1)) * self.signal
        self.loca = np.concatenate((parx, pary, pari), axis=1)
        self.loca = self.nextRandomStep()
        return self.loca

    def genBG(self):
        # generates a quite regular background image
        x = np.arange(self.xfov)
        y = np.arange(self.yfov)
        X, Y = np.meshgrid(y, x)
        simbg = 5*self.noise+np.sin((X+Y)/self.psize)
        return simbg

    def genImage(self):
        """
        :return: generated image with specified noise and particles position in self.loca
        """
        simimage = np.random.uniform(1, self.noise, size=(self.xfov, self.yfov)) + self.bg
        psize = self.psize
        normpar = np.zeros((2*psize, 2*psize))
        for x in range(psize):
            for y in range (psize):
                r = 2*(x**2+6*y**2)/psize**2
                normpar[psize-x, psize-y] = normpar[psize-x, psize+y] = \
                normpar[psize+x, psize-y] = normpar[psize+x, psize+y] = np.exp(-r)
        for n in range(0, self.numpar):
            x = np.int(self.loca[n,0])
            y = np.int(self.loca[n,1])
            simimage[x-psize:x+psize, y-psize:y+psize] = simimage[x-psize:x+psize, y-psize:y+psize] + normpar * self.loca[n,2]

        return simimage

    def nextRandomStep(self):
        numpar = self.numpar
        margin = 2*self.psize #margines for keeping whole particle spread inside the frame
        dr = np.random.normal(loc=0.0, scale=np.sqrt(self.difcon), size=(numpar, 2))
        locations = self.loca[:,0:2] + dr
        for n in range(0, numpar): # checking if particles get close to the margin
            for i in [0, 1]:
                if (locations[n, i] < margin) or (locations[n, i] > self.xfov-margin):
                    locations[n, i] = locations[n, i] - 2*dr[n, i] # applying mirror b.c.

        self.loca[:,0:2] = locations  #particle intensities are not varied between frames, only their locations
        simimage = self.genImage()
        return simimage

