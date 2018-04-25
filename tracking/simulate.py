"""
    CETgraph.tracking.simulate.py
    ==================================
    Contains verious classes for generating synthetic images corresponding to particles performing a thermal Brownian
    motion to be viewed, and eventually analyzed when necessary.
    These classes can be used for example to test the reliability of the tracking algorithms

    .. lastedit:: 16/11/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np
import math

class Waterfall:
    def __init__(self):
        print("Class Waterfall has been renamed to class Kymograph.")

class Kymograph:
    """Generates z-position vs time for a group of particles as they go with
    normal Brownian motion and drift.

    Parameters
    ----------
    fov: 1D field of view in pixels
    numpar: number of particles in the field of view,
    difcon: diffusion constant (identical particles) [pixel^2/frame]

    psize: particle size in pixels
    signal: brightness of each particle
    noise: background random noise
    drift: average drift velosity [pixel/frame]
    Returns
    -------
    float numpy array of intensity(position,time)
    """

    def __init__(self, fov = 500, numpar = 4, nframes = 100, difcon = 1, signal = 10, noise = 1, psize = 8, drift = 1):
        self.fov = fov
        self.difcon = difcon
        self.drift = drift
        self.numpar = numpar
        self.signal = signal
        self.noise = noise
        self.psize = psize
        self.nframes = nframes # number of lines (frames) to be generated
        self.tracks = np.zeros((numpar*nframes,5)) #array with all actual particle coordinates in the format [0-'tag', 1-'t', 2-'mass', 3-'z', 4-'width'] prior to adding noise

    def genKymograph(self):
        numpar = self.numpar
        nframes = self.nframes
        fov = self.fov
        positions = 0.8 * fov * (np.random.rand(numpar) + 0.1) # additional factors for making sure particles are generated not to close to the two ends
        kg = np.zeros((fov, nframes))
        taxis = np.arange(nframes)
        p_tag = 0
        for p in positions:  # generating random-walk assuming dt=1
            steps = np.random.standard_normal(self.nframes)
            path = p + np.cumsum(steps) * np.sqrt(2 * self.difcon) + self.drift * taxis
            intpath = np.mod(np.asarray(path, dtype=int), fov)
            kg[[intpath, taxis]] += self.signal * (1 + p_tag / 10)
            # nest few lines to fill in tracks in the format suitable for analysis
            p_tag += 1
            tags = np.array([((0*taxis)+1)*p_tag])
            masses = tags / p_tag * self.signal * (0.9 + p_tag / 10)
            widths = tags / p_tag * self.psize
            trackspart = np.concatenate((tags, [taxis], masses, [path], widths), axis=0)
            self.tracks[(p_tag-1)*nframes:p_tag*nframes,:] = np.transpose(trackspart)

        fft_tracks = np.fft.rfft2(kg, axes=(-2,))
        max_freq = int(self.fov / self.psize)
        fft_tracks[max_freq:, :] = 0
        kg = abs(np.fft.irfft2(fft_tracks, axes=(-2,)))
        noise = np.random.randn(self.fov, self.nframes)
        kg += noise
        return kg


class SingleFrame:
    """
    :param size: [width, height] of the desired image that contains these particles
    :return: SingleFrame.loca: intended location of the particles (with sub-pixel resolution)
             SingleFrame.genImage: an image with specified noise and particles displaced accordingly
    """
    def __init__(self, fov = [300, 200], particleDiameter = 100,temperature = None,viscosity = None,numpar = None, difcon = None, signal = 10, noise = 1, psize = 8, useRandomIntensity = True, numberPerSpecies = [None], signalPerSpecies = [None], pixelSizePerSpecies = [None], staticNoise = 0.0,FPS = None, micronPerPixel = None):
        # camera and monitor parameters
        self.xfov, self.yfov = fov
        # simulation parameters
        boltzmannConstant = 1.380649 #10^-23 /K
        if(temperature == None):
            temperature = 293  #K
        if(viscosity == None):
            viscosity = 0.001 #Pa s
        if(difcon == None):
            difcon = (boltzmannConstant*temperature/(3*math.pi*viscosity*particleDiameter*FPS*micronPerPixel*micronPerPixel*100)) #the 100 is because boltsmann /(micron/pixel)^2*diameter 10^-23 * / (10^-12*10^-9) = 10^-2

        print("Diffusion constant set on " + str(difcon) + "  pixel^2 per frame  "+ str(difcon*FPS*micronPerPixel*micronPerPixel) + "  micron^2 /s")
        
        
        self.difcon = difcon # Desired diffusion constant in pixel squared per frame
        self.numpar = numpar # Desired number of diffusing particles
        self.signal = signal # brightness for each particle
        self.noise = noise # background noise
        self.psize = psize # half-spread of each particle in the image, currently must be integer
        self.useRandomIntensity = useRandomIntensity # use a random intensity or a fixed intensity per particle
        self.numSpecies = len(numberPerSpecies) #number of different species  (i.e. number of different intensities)
        self.staticNoise = staticNoise # white noise to chane each frame
        
        
        
        if(numberPerSpecies == [None]):
            self.numberPerSpecies = [numpar]
        else:
            self.numberPerSpecies = numberPerSpecies
            
        if(pixelSizePerSpecies == [None]):
            self.pixelSizePerSpecies = [psize]
        else:
            self.pixelSizePerSpecies  = pixelSizePerSpecies         
        

        self.psize = np.int(max(max(pixelSizePerSpecies),max([8,psize])))
        
        
        if(4*self.psize >= self.xfov or 4*self.psize >= self.yfov):
            print("Particle size too large for field of view.")

        if(signalPerSpecies == [None]):
            self.signalPerSpecies = [signal]
        else:
            self.signalPerSpecies = signalPerSpecies
            
        if(numpar == None):
            if(numberPerSpecies == [None] ):
                self.numpar = 4 #old default value
            else:
                self.numpar =np.sum(self.numberPerSpecies)
        
        if (self.numpar != np.sum(self.numberPerSpecies)):
            print("Sum of particles per species != number of particles. Using sum of species as number of particles instead.\n")
            print(self.numpar)
            print(self.numberPerSpecies)
            self.numpar = np.sum(numberPerSpecies)
        if (len(signalPerSpecies) != len(signalPerSpecies)):
            print("Array lengths of number per species and different signal per species do not match")
        self.initPSF()
        self.bg = self.genBG()
        self.initLocations()

    def addWhiteNoise(self, frame = [[],[]], whiteNoise = 0.0):
        """
        Add white noise each step, so the noise will differ each frame.
        """
        if(whiteNoise > 0):
            for i in range(self.xfov):
                for j in range(self.yfov):
                    frame[i][j] = np.int8(frame[i][j] + np.random.uniform(0, whiteNoise*255.0))
        return frame
        
    def initPSF(self):
        psize = self.psize
        normpar = np.zeros((self.numpar,2*psize, 2*psize))
        pixelSizePerParticle = []
        
        for i in range(self.numSpecies):
            pixelSizePerParticle.extend(np.full(self.numberPerSpecies[i], self.pixelSizePerSpecies[i]))
        
        
        for n in range(self.numpar):
            for x in range(psize):
                for y in range (psize):
                    r = 2*(x**2+y**2)/pixelSizePerParticle[n]**2
                    normpar[n][psize-x, psize-y] = normpar[n][psize-x, psize+y] = \
                    normpar[n][psize+x, psize-y] = normpar[n][psize+x, psize+y] = np.exp(-r)
        self.normpar = normpar
        

    def initLocations(self):
        # initializes the random location of numpar particles in the frame. one can add more paramaters like intensity
        # and PSF distribution if necessary
        parx = np.random.uniform(0, self.xfov, size=(self.numpar, 1))
        pary = np.random.uniform(0, self.yfov, size=(self.numpar, 1))
        pari = []
        if (self.useRandomIntensity):
            for i in range(self.numSpecies):
                pari.extend(np.random.uniform(1, self.numberPerSpecies[i], size=(self.numberPerSpecies[i], 1)) * self.signalPerSpecies[i])
        else:
            for i in range(self.numSpecies):
               pari.extend(np.full((self.numberPerSpecies[i],1), self.signalPerSpecies[i]))
               
        self.loca = np.concatenate((parx, pary, pari), axis=1)
        self.loca = self.nextRandomStep()
        return self.loca

    def genBG(self):
        # generates a quite regular background image
        x = np.arange(self.xfov)
        y = np.arange(self.yfov)
        X, Y = np.meshgrid(y, x)
        simbg = 5*self.noise+np.sin((X+Y)/self.psize)
        simbg = self.addWhiteNoise(simbg,self.staticNoise)
        return simbg

    def genImage(self):
        """
        :return: generated image with specified noise and particles position in self.loca
        """
        simimage = np.random.uniform(1, self.noise, size=(self.xfov, self.yfov)) + self.bg
        psize = self.psize

        for n in range(self.numpar):
            x = np.int(self.loca[n,0])
            y = np.int(self.loca[n,1])
            simimage[x-psize:x+psize, y-psize:y+psize] = simimage[x-psize:x+psize, y-psize:y+psize] + self.normpar[n] * self.loca[n,2]

        return simimage


    def nextRandomStep(self):
        numpar = self.numpar
        margin = 2*self.psize #margines for keeping whole particle spread inside the frame
        dr = np.random.normal(loc=0.0, scale=np.sqrt(2*self.difcon), size=(numpar, 2))#must change this to 2sqrt?
        locations = self.loca[:,0:2] + dr
        for n in range(numpar):
            locations[n, 0] = np.mod(locations[n, 0]-margin, self.xfov-2*margin)+margin
            locations[n, 1] = np.mod(locations[n, 1]-margin, self.yfov-2*margin)+margin
        self.loca[:,0:2] = locations
        return self.loca


    def genStack(self, nframes=100):
        """
        Using all the above methods in this class, this method only iterates enough to create a stack of synthetic frames
        that can be analyzed later

        :param nframes: number of frames to generate
        :return: numpy array of nframes stack of brownian motion of particles with additional noise
        """
        numpar = self.numpar
        data = np.zeros((self.xfov, self.yfov, nframes))
        # tracks = np.zeros((nframes*numpar,4)) #in next version it is good to return also the actual locations next to the image stack
        for i in range(nframes):
            l = self.nextRandomStep()
            data[:,:,i] = self.genImage()
            #if i%int(nframes/10) == 0:
                #print(str(int(i*100/nframes)) + "%")
        return data

