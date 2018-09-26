# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:32:33 2018

@author: Peter
"""

"""

This class provides code to track particles in hdf5 data format arrays. This
class works together with trackUtils for importing and exporting data. Because 
of the large number of parameters of the output of these functions, the output
will be variables of the object that belongs to this class. That object can be
feeded to functions of trackUtils.


subtractBackground
optional arguments: background = []
This is an background image that will be subtracted from all frames. If this is
left empty, a frame from the frame stack will be subtrackged.

detectParticles
Detects the particles with the particles given to the Tracking object. ít does
not return the particles, but stores it in the global variable particles.

linkParticles
This function links particles detected by detectParticles


filterLinks
This function filters links based on parameters given to the object.

calculateMobility
The function calculates the mobility of the particles in either the x or the y 
direction.

calculateDiffusion
This function calculates the diffusion and diffusion based output.
"""

import datetime  # date stamp
from collections import Counter  # used to count #frames in which individual particles occur: used for weighted fit
import math  # maths utils
import matplotlib.pyplot as plt# plotting
import numpy as np  # numpy arrays
import trackpy as tp  # trackpy
from trackUtils import trackUtils


class Tracking:

    def __init__(self, folder, particleDiameter, minimumMass, maximumMass,
                 frameMemory, micronPerPixel, dataKey=None, h5name="data.h5", 
                 FPS=-1, useFrames=-1, subframes=[None, None], signalType=None,
                 electricFrequency=-1):

        """
       
          A new Tracking object requires a folder, particle diameter in px, 
          minimum mass, maximum mass, maximum number of frames a particle can be
          out of focus or FOV and micron per pixel.
          The optional parameters are: datakey, the name of the datakey of the 
          hdf5 file. This will be redundant by moving the import functions to 
          trackUtils py.
          h5name is the name of the raw data. This will also be redundant. 
          FPS is the frames per second of the recorded images. 
          subframes is the window of frames that will be selected. This uses
          all frames by default.
          signalType is a string that describes the wave form of the current in
          measurements on electrophoretic mobility. 
          electricFrequency is the frequency of the current. 
        
        
        The Tracking class contains all tracking functions. The folder of the 
        tracking objet from which the metadata will be loaded from two different
       files, instructions.txt and metadata.txt. The will be moved to trackUtils.
       
       


        """
        # mpl.use('Agg')

        print("Initialize parameters")
        print(minimumMass)
        if (minimumMass < 0):
            minimumMass = int(trackUtils.loadFromInstructionFile("Minimum mass", folder + "\\instructions.txt"))
            print("Minimum mass: " + str(minimumMass))
        if (maximumMass < 0):
            maximumMass = int(trackUtils.loadFromInstructionFile("Maximum mass", folder + "\\instructions.txt"))
            print("Maximum mass: " + str(maximumMass))
        if (particleDiameter < 0):
            particleDiameter = int(trackUtils.loadFromInstructionFile("Diameter", folder + "\\instructions.txt"))
            print("Particle diameter (px): " + str(particleDiameter))
        if (signalType == None):
            self.signalType = trackUtils.getFromMetaData("SignalType", (folder + "\\metadata.txt"))
            print("Electric signal: " + str(self.signalType))

        self.subframes = subframes
        if (subframes != [None, None]):
            self.useFrames = np.amax(subframes) - np.amin(subframes)
        self.currentPath = folder  # path to data folder
        self.framePath = self.currentPath + "\\" + h5name  # path to hdf5 file
        self.removeBackground = False  # subtractBG or not (for metadatafile)
        self.particleDiameter = particleDiameter  # particle diameter in pixels
        self.minMass = minimumMass  # minimal mass particle
        self.maxMass = maximumMass  # maximum mass particle (intensity)
        self.maxTravelDistance = 10  # maximum distance travelled between frames
        self.minimumMSD = 0.05  # minimum mean square displacement for calculation D, because D is calculated for each particle individually, this parameter is not important
        self.maxFrameMemory = frameMemory  # If particle disappears, it will be a new unique particle after this number of frames
        self.useFrames = useFrames  # if < 0, all frames will be used.
        self.minimumParticleLifetime = 10  # minimum number of frames a particle should be located. Otherwise it is deleted
        self.cameraFPS = FPS  # camera FPS if < 0, script will try to get from metadata file
        self.micronPerPixel = micronPerPixel  # pixel size in um
        self.removeBackgroundOffset = 50  # subtracts the image n frames before the current frame to remove bg
        self.today = datetime.datetime.now().strftime("%y-%m-%d")  # get date
        self.temperature = 293  # temp in K
        self.boltzmannConstant = 1.380649  # (in 10^-23 JK^-1)
        self.viscosity = 0.001  # viscosity in Pa s
        self.runs = 0  # will be set depending on folder names
        self.maximumEccentricity = 0.5  # maximum eccentricity of detected particles  
        self.frames = trackUtils.loadData(self.framePath, self.subframes,
                                          dataKey=dataKey)  # will be set by createDirectoryTree()
        self.frameDimensions = self.frames.shape  # Will be set by loadData 
        self.particles = []  # detected particles, will be filled by detectParticles()
        self.numberOfFramesFitted = -1  # if <0 all frames will be used.
        self.maxLagTime = 5  # maximum number of frames used for msd calculation
        self.removeLastFrames = False  # remove last frames to prevent artefacts from background removal
        self.window = 0.3  # window for Fourier filter
        self.electricFrequency = electricFrequency

        if (self.electricFrequency < 0 or self.electricFrequency == None):
            self.electricFrequency = float(
                trackUtils.getFromMetaData("ElectricFrequency", (self.currentPath + "\\metadata.txt")))
        self.maxFrames = len(self.frames)

        # some sanity checks:

        if (self.useFrames >= 0 and self.useFrames < self.maxFrames):
            self.maxFrames = self.useFrames
        if self.useFrames > len(self.frames):
            print(
                    "Maximum number of frames larger than number of imported frames: using number of imported frames instead. \nRequested number of frames: " + str(
                self.useFrames) + " Number of frames in data: " + str(len(self.frames)))

        if (self.maxFrames < self.minimumParticleLifetime):
            print(
                "useFrames < minimumParticleLifetime. This means all particles will be deleted and no tracking will be done.")

        if (self.micronPerPixel < 0):
            self.micronPerPixel = float(trackUtils.getFromMetaData("PixelSize", (self.currentPath + "\\metadata.txt")))

        if (self.cameraFPS < 0):
            self.cameraFPS = float(
                trackUtils.getFromMetaData("ResultingFrameRate", (self.currentPath + "\\metadata.txt")))
            self.exposureTime = 1000000.0 / self.cameraFPS
            if (self.cameraFPS < 0):
                print("Could not find FPS. Set exposure time to 40 FPS")
                self.exposureTime = 1000000.0 / 40
                self.cameraFPS = 40
        else:
            self.exposureTime = 1000000.0 / self.cameraFPS
        print("FPS is: " + str(self.cameraFPS))

    def subtractBackground(self, silent=True, background=[]):
        if (background == []):
            print("subtracting background by subtracting a previous frame")
        else:
            print("Subtracking calculated background")

        self.removeBackground = True
        frames = []  # this will be the frames without bg
        frames0 = self.frames.astype(np.int16)  # negative values are needed, negative values are set to 0 by clip
        if (background == []):
            maxFrames = np.amin([self.maxFrames + self.removeBackgroundOffset, len(frames0)])
            for i in range(maxFrames):
                index = np.mod(i + self.removeBackgroundOffset, len(frames0))
                if ((i % int(len(frames0) / 10) == 0) and not silent):
                    print(str(int(i * 100 / len(frames0))) + " %")
                if index < 0:
                    index = len(frames0) + index  # beware '+' because index < 0
                frames.append(np.subtract(frames0[i], frames0[index]))

        else:
            if (background.shape != frames0[0].shape):
                print("Wrong background shape: ")
                print(background.shape)
                print(frames0[0].shape)
            maxFrames = len(frames0)
            for i in range(maxFrames):
                frames.append(np.subtract(frames0[i], background))

        frames = np.array(frames)
        # bring each pixel value below 0 back to 0:
        np.clip(frames, 0, np.amax(frames), frames)
        frames = np.uint16(frames)
        if (self.removeLastFrames):
            self.frames = frames[:-self.removeBackgroundOffset]
            self.frameDimensions = frames.shape
        else:
            self.frames = frames

        return frames

    def getSettings(self, dest="", writeOutput=True):
        """
        This function writes different parameters to a metadatafile. will be 
        moved to trackUtils.
        """

        metadataText = "Run " + str(self.runs) + ' ' + str(datetime.datetime.now()) + \
                       "\n" + "Using data from file: " + self.framePath + "\n\nSettings:\n" + \
                       "particle diameter (px): " + str(self.particleDiameter) + "\n" + \
                       "particle diameter (bottle): " + str(self.estimatedDiameter) + "\n" + \
                       "minimum mass: " + str(self.minMass) + "\n" + \
                       "maximum mass: " + str(self.maxMass) + "\n" + \
                       "maximum eccentricity: " + str(self.maximumEccentricity) + "\n" + \
                       "maximum distance between frames (px): " + str(self.maxTravelDistance) + "\n" + \
                       "minimum distance between frames (px): " + str(self.minimumMSD) + "\n" + \
                       "memory (frames): " + str(self.maxFrameMemory) + "\n" + \
                       "minimum number of frames of particle trajectory (frames): " + str(
            self.minimumParticleLifetime) + "\n" + \
                       "max lag time ( rames): " + str(self.maxLagTime) + \
                       "number of frames used (frames): " + str(self.useFrames) + "\n" + \
                       "number of frames of full data (frames): " + str(self.frameDimensions[0]) + "\n" + \
                       "Image size: " + str(self.frameDimensions[1]) + 'x' + str(self.frameDimensions[2]) + "\n" + \
                       "FPS: " + str(self.cameraFPS) + "\n" + \
                       "Exposure time (us): " + str(self.exposureTime) + "\n" + \
                       "Micron/pixel : " + str(self.micronPerPixel) + "\n" + \
                       "Background subtracted: " + str(self.removeBackground) + "\n"

        if (dest != "" and writeOutput):
            self.writeMetadata(metadataText, dest)
        elif (dest == "" and writeOutput):
            self.writeMetadata(metadataText, self.currentPath + "/metadata.txt")
        return metadataText

    def linkParticles(self, filtering=True, silent=False):
        """This function links the outut of detectParticles. """ 
        self.links = tp.link_df(self.particles, self.maxTravelDistance, memory=self.maxFrameMemory)
        if (filtering):
            self.links = self.filterLinks(silent=silent)
        return self.links

    def filterLinks(self, silent=False):
        
        
        print("Filtering.")
        links = self.links
        print("Number of links: " + str(len(links)))
        
        links = tp.filter_stubs(links, threshold=self.minimumParticleLifetime)
        print("Number of links after removing paths shorter than %d: %d " % (self.minimumParticleLifetime, len(links)))
        links = links[((links['mass'] < self.maxMass) & (links['ecc'] < self.maximumEccentricity))]
        print("Number of links after removing high intensity and particle with high eccentricity: %d " % len(links))
        self.linksWithoutDriftCorrection = links

        self.drift = tp.compute_drift(links)
        links = tp.subtract_drift(links.copy(), self.drift)
        if (not silent):
            """
            plot the mass versus size and plot the trajectories of filtered links:
            """
            plt.figure()
            figure = plt.gcf()
            tp.mass_size(links.groupby('particle').mean())
            figure.savefig(self.currentPath + '/massSizePlot' + str(self.runs) + '.pdf')
            # plt.show(figure)

            plt.figure()
            figure = plt.gcf()
            tp.plot_traj(links);
            figure.savefig(self.currentPath + '/trajectories' + str(self.runs) + '.pdf')
            # plt.show(figure)
        print("Number of links: " + str(len(links)))
        self.links = links

        return links

    def _getPointsWithLargestDistance(self, xValues, yValues):
       
       if(len(xValues) != len(yValues)):
          #print("Points wrong shape")
          return
       if(len(xValues) < 2):
          #print("Too few points in array")
          return
       
       meanX = np.mean(xValues)
       meanY = np.mean(yValues)  
       p0 = [None, None]
       p1 = [None, None]
    
       distance2 = 0.0
       for x, y in zip(xValues, yValues):
          if((x-meanX)*(x-meanX)+(y-meanY)*(y-meanY) > distance2):
             distance2 = (x-meanX)*(x-meanX)+(y-meanY)*(y-meanY)
             p0[0], p0[1] = [x, y]
             
       distance2 = 0.0
       for x, y in zip(xValues, yValues):
          if((x-p0[0])*(x-p0[0])+(y-p0[1])*(y-p0[1]) > distance2):
             distance2 = (x-p0[0])*(x-p0[0])+(y-p0[1])*(y-p0[1])
             p1[0], p1[1] = [x, y]
             
       return (np.sqrt(distance2),p0,p1) 


    def calculateMobility(self, direction='y', useFFT=True, frequency=None, window=None, signal=None):
        """The mobility will be calculated in a direction specified by the 
        optional parmeter direcction. The frequency  of  the current and the is
        the optional parameter frequency. The width of the fourier filter is 
        window. This is in Hz. signal is a string that is either 'sin' or 'block'
        This changes the normalisation of the calculated mobility.
        """
        if (window == None):
            window = self.window
        else:
            self.window = window

        if (frequency == None):
            frequency = self.electricFrequency
        else:
            self.electricFrequency = frequency

        if (signal == None):
            signal = self.signalType
            if (isinstance(signal, int)):
                if (signal < 0):
                    signal = "block"
            print("Electric signal " + signal)
        print('Calculating mobility in ' + direction + '-direction')

        drift = tp.compute_drift(self.linksWithoutDriftCorrection)

        particleVelocity = np.diff(drift[direction]) - np.diff(drift[direction]).mean()

        self.particleSpeed = abs(particleVelocity)
        self.mobility = self.particleSpeed.mean() * self.cameraFPS * self.micronPerPixel
        if (signal == "sin" or signal == "cos" or signal == "s"):
            print("Multipy mobility by pi/2 to compensate for sinusoidal signal.")
            self.mobility = self.mobility * math.pi / 2.0

        drift = drift[direction] - drift[direction].mean()
        plt.figure()
        plt.plot(drift)
        plt.show()

        FFTDrift = np.fft.fft(drift) / len(drift)
        frequencies = np.fft.fftfreq(len(FFTDrift), 1 / self.cameraFPS)

        plt.figure()
        plt.plot(frequencies)
        plt.show()

        if (window > 0):
            for i, f in enumerate(frequencies):
                if (np.abs(f) < frequency - window or np.abs(f) > frequency + window):
                    FFTDrift[i] = 0

        self.filteredDrift = np.fft.ifft(len(drift) * FFTDrift)
        FFTDrift = np.abs(FFTDrift)
        
        """
        These plots must be moved to trackUtils.
        """

        plt.figure()
        figure = plt.gcf()
        plt.ylabel('Displacement (px)')
        plt.xlabel('frame');
        plt.plot(self.filteredDrift)
        figure.savefig(self.currentPath + '/FourierFiltedDrift' + str(self.runs) + '.pdf')

        self.amplitudeFFT = math.sqrt(2 * abs(self.filteredDrift * self.filteredDrift).mean())
        self.mobilityFFT = 4 * frequency * self.amplitudeFFT * self.micronPerPixel
        if (signal == "sin" or signal == "cos"):
            self.mobilityFFT = self.mobilityFFT * math.pi / 2.0

        plt.figure()
        figure = plt.gcf()
        plt.ylabel('Amplitude')
        plt.xlabel('frequency (Hz)');
        plt.plot(frequencies, FFTDrift)
        figure.savefig(self.currentPath + '/powerSpectrum' + str(self.runs) + '.pdf')

        plt.figure()
        figure = plt.gcf()
        plt.ylabel('Amplitude')
        plt.xlabel('frequency (Hz)');
        plt.plot(frequencies[:200], FFTDrift[:200])
        figure.savefig(self.currentPath + '/powerSpectrumZoom' + str(self.runs) + '.pdf')

        # calculate mobility for each individual particle.

        links = self.linksWithoutDriftCorrection
        meanDrift = np.diff(tp.compute_drift(links)[direction]).mean()
        yvalues = links[['frame', 'particle', direction]].groupby('particle')[direction].apply(list)
        listOfFrames = links[['frame', 'particle', direction]].groupby('particle')['frame'].apply(list)
        eccentricities = links[['particle', 'ecc']].groupby('particle')['ecc'].apply(list)

        particleVelocity = []
        particleSpeed = []
        self.averageMobilities = []
        framesJump = []
        self.averageMobilityWeights = []
        self.eccentricity = []
        self.eccentricityWeights = []
        
        """
        The eccentricity for each particle, together with the mobility based
        on the speed of the particle calculated from its displacement per 
        frame. The time the partilce is in FOV will be saved as well so it can
        be used as a weight.
        """


        for i in yvalues.index.values:
            framesJump.append(np.diff(listOfFrames[i]))
            particleVelocity.append(np.diff(yvalues[i]) - meanDrift)
            if (self.eccentricity is not []):
                self.eccentricity.append(np.mean(eccentricities[i]))
                self.eccentricityWeights.append(len(eccentricities[i]))
            else:
                self.eccentricity = [np.mean(eccentricities[i])]
                self.eccentricityWeights = [len(eccentricities[i])]

            for j in range(len(particleVelocity[-1])):
                particleVelocity[-1][j] = particleVelocity[-1][j] / framesJump[-1][j]
            if (len(particleVelocity[-1]) > 2):
                particleSpeed.append(abs(particleVelocity[-1]))
                self.averageMobilities.append(particleSpeed[-1].mean() * self.cameraFPS * self.micronPerPixel)
                self.eccentricity
                self.averageMobilityWeights.append(len(particleSpeed[-1]))

        self.averageMobilityWeights = self.averageMobilityWeights / (np.sum(self.averageMobilityWeights))

        if (len(self.averageMobilities) > 2):
            plt.figure()
            n, bins, patches = plt.hist(self.averageMobilities, 35, weights=self.averageMobilityWeights)
            figure = plt.gcf()
            plt.xlabel('Mobility ($\mu$m/s)')
            plt.ylabel('Count')
            plt.title('Histogram of Mobility')
            plt.grid(True)
            figure.savefig(self.currentPath + '/mobilityDistribution' + str(self.runs) + '.pdf')

        """
        For short tracks, the begin and end point of the track are the points
        with the largest distance between them. The distance between those
        points is the length of the track and the distance covered by the
        particle.
        """
        xValues = self.linksWithoutDriftCorrection[['particle', 'x']].groupby('particle')['x'].apply(list)
        yValues = self.linksWithoutDriftCorrection[['particle', 'y']].groupby('particle')['y'].apply(list)
        
      
        
        self.outerPointList = []
        self.outerPointDistance = []
        self.outerPointDistanceWeights = []
        self.mobilityPerTrackList = []
        outerPoints = [0,0,0,0]
        for i in yValues.index.values:
           temp = self._getPointsWithLargestDistance(xValues[i],yValues[i])
           outerPoints = [temp[1][0],temp[1][1],temp[2][0],temp[2][1]]
           if(outerPoints == None):
              continue
           self.outerPointList.append(outerPoints)
           self.outerPointDistance.append(temp[0])
           self.mobilityPerTrackList.append(math.pi*frequency*self.micronPerPixel*temp[0])
           self.outerPointDistanceWeights.append(len(xValues[i]))
        return
    

    def detectParticles(self, maxFrames=0, silent=False):
        """This function looks for particles in a frame stack."""
        if (maxFrames == 0):
            maxFrames = self.maxFrames
        self.particles = tp.batch(self.frames[:maxFrames], self.particleDiameter, minmass=self.minMass)
        if (not silent):
            fig, ax = plt.subplots()
            ax.hist(self.particles['mass'], bins=20)
            ax.set(xlabel='mass', ylabel='count')
            plt.savefig(self.currentPath + '/massDistribution_run' + str(self.runs) + '.pdf')
        return

    def showDetectedParticles(self, frameNumber=0, path="", saveImages=True, showImages=True, annotatedFrame=False,
                              invert=False, logarithm=True):
        """
        function for diagnostic purposes: it outputs an annotated image of detected particles and some 
        plots. It outputs a mass distribution of a single frame (as specified in header) and the decimal 
        value of estimated x and y positions of found particles. If This distribution is not homogeneously 
        distributed, noise is too ofted detected as particle.

        """
        if (path == ""):
            path = self.currentPath

        particles = tp.locate(self.frames[frameNumber], self.particleDiameter, minmass=self.minMass)
        particles = particles[((particles['mass'] < self.maxMass) & (particles['ecc'] < self.maximumEccentricity))]
        if (len(particles) == 0):
            print("No particles found.")
            return particles

        print('Minimum mass: ' + str(np.amin(np.array(particles['mass']))))
        print('Maximum mass: ' + str(np.amax(np.array(particles['mass']))))
        print('Number of particles: '+ str(len(particles['mass']) ))

        if (showImages):
            plt.figure()
            figure = plt.gcf()
            if (annotatedFrame):
                if (logarithm):
                    logframes = np.log(self.frames[frameNumber] + 1)
                    maxPixelValue = np.amax(logframes)
                    logframes = (1.0 * logframes / maxPixelValue) * 65535.0
                    logframes = logframes.astype(np.uint16)
                    tp.annotate(particles, logframes, invert=invert)
                else:
                    tp.annotate(particles, self.frames[frameNumber], invert=invert)
            if (saveImages):
                figure.savefig(path + '/detectedParticles_run' + str(self.runs) + "_frame_" + str(frameNumber) + '.png')
            # plt.show(figure)
            if (not annotatedFrame):
                fig, ax = plt.subplots()
                ax.hist(particles['mass'], bins=20)
                ax.set(xlabel='mass', ylabel='count')
                if (saveImages):
                    plt.savefig(path + '/massDistribution_run' + str(self.runs) + "_frame_" + str(frameNumber) + '.pdf')

                plt.figure()
                figure = plt.gcf()
                tp.subpx_bias(particles)
                if (saveImages):
                    plt.savefig(path + '/subPixelBias_run' + str(self.runs) + "_frame_" + str(frameNumber) + '.pdf')
                # plt.show(figure)
        return particles

    def calculateDiffusion(self, maxLagTime=None):

        """ 
        The diffusion is calculated by slicing the data in slices of length maxLagTime and fitting 
        the mean square displacement to (At)^n. n should be approximately 1 and A  = 4D.

        This function also stores the fitted data (fitted power, diffusion constant, particle diameter)
        per particle in different lists for future use. 
        """
        print("Calculating Diffusion")

        if (maxLagTime == None):
            maxLagTime = self.maxLagTime
        else:
            self.maxLagTime = maxLagTime

        self.msdData = tp.imsd(self.links, self.micronPerPixel, self.cameraFPS, max_lagtime=maxLagTime)
        self.fits = []
        self.diffusionConstants = []
        self.particleDiameters = []
        self.fittedPower = []
        self.visibleInFrames = []
        self.averageMass = []
        # deletes null and nan values, and minimumMSD check:
        occurance = Counter(self.links['particle'])  # Count in how many frames a particle is seen. (occurance[particle number] gives its path length)
        print("Number of particles: " + str(len(occurance)))
        massList = self.links[['particle', 'mass']].groupby(['particle']).mean()
        """ This loop loops over each particle and if the data of that particle
        is finite, it will be added t the list. on that data the diffusion
        constant will be fitted. The mass of the particle will be stored in 
        a list as well"""
        for i in self.msdData.iloc[0, :].index.values:  # loop over each particle number
            temp = self.msdData[i][np.isfinite(self.msdData[i]) & self.msdData[i] > self.minimumMSD]
            if (len(temp) < 2):
                continue
            temp2 = [temp.iloc[0]] + np.diff(temp).tolist()
            temp2 = sum(temp2) / (float(len(temp2)))
            if ((not temp.isnull().values.any()) and (len(temp) > 0) and (temp2 > self.minimumMSD)):
                self.fits.append([i, tp.utils.fit_powerlaw(temp, plot=False).as_matrix().tolist()])
                if ((not np.isnan(self.fits[-1][1][0][0])) and (not np.isnan(self.fits[-1][1][0][1]))):
                    self.visibleInFrames.append(occurance[i])  # number of frames a particle is seen is used to give weigths to particle diamater histogram
                    self.averageMass.append(massList.loc[i])
        """From the fits, the diffusion constant and the diameter is calculated. """
        for e in self.fits:
            if ((not np.isnan(e[1][0][0])) and (not np.isnan(e[1][0][1]))):
                self.fittedPower.extend([e[1][0][0]])
                self.diffusionConstants.append([e[0], 0.25 * e[1][0][1]])  # 0.25 because of a == 4D
                self.particleDiameters.append([e[0], (4 * self.boltzmannConstant * self.temperature) / (
                        3 * math.pi * self.viscosity * 100 * e[1][0][1])])
        self.diffusionConstants = np.array(self.diffusionConstants)
        self.particleDiameters = np.array(self.particleDiameters)
        if (self.diffusionConstants == []):
            print("Problem with diffusion constant.\n")
            print(self.diffusionConstants)

        if (maxLagTime == None):
            maxLagTime = self.maxLagTime
        else:
            self.maxLagTime = maxLagTime
        powerLaw = tp.emsd(self.links, self.micronPerPixel, self.cameraFPS, max_lagtime=maxLagTime)
        trackpyFit = tp.utils.fit_powerlaw(powerLaw)
        print(str(trackpyFit))
        self.trackpyFit = trackpyFit.as_matrix()
        self.retrievedParticleDiameter = (4 * self.boltzmannConstant * self.temperature) / (
                3 * math.pi * self.viscosity * 100 * self.trackpyFit[0][1])

        return
