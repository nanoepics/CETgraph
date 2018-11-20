"""
   CETgraph.tracking.simulateBrownianMotion.py
   ==================================
   This class simulates image stacks of brownian particles. This simulation is
   based on the method described by Savin and Doyle [1].
   
   
   [1] T Savin and P. Doyle, Static and Dynamic Errors in Paticle Tracking
   Microrheology, Biophysical Journal, 88, 623-638, 2005.

    .. lastedit:: 25-09-2018
    .. sectionauthor:: Peter Speets <p.n.a.speets@students.uu.nl>
"""


import numpy as np
import math
import datetime

class Particle:
    
    #This class mimics a C-style struct. It has no functions, but holds the
    #parameters of a particle. diameter and diffusionConstant are stored as
    #different numbers, but one should follow from the other.
    
    #particleCount is to give a particle an ID each time the object is created
    
    _particleCount = 0
    
    def __init__(self, 
                 pos, diameter, diffusionConstant, intensity, sigma,
                 mobility = 0.0, PSFSize = (10,10), species = 1 ):
        #The resulting PSF will be intensity*exp(2*r^2/sigma^2) with r the
        #distance from pos.
        
        self.particleID = Particle._particleCount
        self.species = species #int for later reference
        self.pos = pos # (x, y) position in px. These are floats
        self.diameter = diameter #diameter in nm, for Brownian displacement
        self.sigma = sigma #in px^2
        self.intensity = intensity #float, but will be pixel value
        self.PSFSize = PSFSize #in px
        self.mobility = mobility #in pixel DELTAT^-1 field^-1
        self.diffusionConstant = diffusionConstant # in px^2/DELTAT
        
        #A new particle is added, so increase Particle.particlecount
        Particle._particleCount += 1
        print("Number of particles: %d" % Particle._particleCount)
        
        
class BrownianSimulation:
    
    def __init__(self, FOV, deltat, micronPerPixel, 
                 temperature = 293, viscosity = 0.001, margin = 20,
                 noiseBackLevel = 0.0, gaussianNoiseLevel = 0.0,
                 electricField = [0.0, 0.0],
                 electricFrequency = -1.0,
                 electricSignalType = 'b'):
        """
        TODO: docstring
        """
        #constants:
        self.FOV = FOV  #field of view in pixels
        self.DELTAT = deltat # delta t in s.
        self.MICRONPERPIXEL = micronPerPixel
        self.TEMPERATURE = temperature # in K
        self.VISCOSITY = viscosity # in Pa s
        self.MARGIN = margin #prevents half particles at the boundary
        self.BOLTZMANNCONSTANT = 1.380649  # in 10^-23 /K
        self.stepNumber = 0
        self.noiseBackLevel = noiseBackLevel #in pixel value.
        self.gaussianNoiseLevel = gaussianNoiseLevel # in pixel value
        self.cameraOn = False
        self.electricAmplitude = electricField.copy()
        self.electricFrequency = electricFrequency
        
        #the three values below are the counters set by simulateBatch. If the 
        #simulation is run without simulateBatch with nextStep function only, 
        #these variables need to be set explicitly in the BrownianSimulation
        #object
        #
        self.maxFrames = 0
        self.stepsPerFrame = 0
        self.stepsExposed = 0

        if(electricSignalType == 's'):
            self.electricField = [0.0, 0.0]
        else:
            self.electricField = electricField.copy()
       
        #self.particles contains particle objects. addParticle adds particles
        #in this list. removeParticle removes particles from this list. 
        #add 
        self.particles = []
        
        #fill an array with zeroes as background. Can be filled with 
        #addBackground()
        self.BACKGROUND = np.zeros(FOV)
        self.image = np.zeros(FOV)
        
        #frames a list of floats that contain the simulated data frames. The 
        #function simulateBatch() appends the data on this list. Bin
        #these floats into ints to get data similar to imaged data. 
        #tracks contains the real positions of the particles so it can be 
        #compared with the tracked positions.
        
        self.frames = []
        self.tracks = []
        
    def addParticle(self, diameter, intensity, sigma, **kwargs):
        #The diffusion constant is calculated from the Einstein diffusion
        #equation D  = k_B T / (3 pi eta d)
        #the unit is in px**2/DELTAT, 

        pos = [(self.FOV[0]-2*self.MARGIN)*np.random.rand() + self.MARGIN,
              (self.FOV[1]-2*self.MARGIN)*np.random.rand() + self.MARGIN]
        
        diffusionConstant = (
                (self.DELTAT*self.BOLTZMANNCONSTANT*self.TEMPERATURE)
                /(
                    3*math.pi
                    *self.VISCOSITY
                    *diameter
                    *self.MICRONPERPIXEL
                    *self.MICRONPERPIXEL
                    *100)
                ) 
        
        self.particles.append(
                Particle(pos, diameter, diffusionConstant,
                         intensity, sigma, **kwargs))
        
        return
        
        
    def removeParticle(self, particleNumber):
        if(particleNumber >= len(self.particles) or particleNumber < 0):
            print("Cannot remove: particle does not exist.")
            return
        self.particles.pop(particleNumber)
        
    def generateNewImage(self):
        #start with background.
        self.image = self.BACKGROUND.copy()
        return self._addStepToImage()
    
    def _addStepToImage(self):
        for particle in self.particles:
            self._drawParticle(particle)
        return self.image
    
    def addGaussianNoise(self, frame, mu, sigma):
        if(sigma <= 0.0 and mu == 0.0):
            return frame
        frame += np.random.normal(loc = mu, scale=sigma, size = frame.shape)
        return frame
    
    
    def _drawParticle(self, particle):
        #This function adds the PSF of the particle to the image array.


        #check boundary FOV. The particle will be drawn in a rectangle from x0
        #to x1 and from y0 to y1.

        x0 = np.amax([0, int(particle.pos[0]) - particle.PSFSize[0]])
        x1 = np.amin([self.FOV[0], int(particle.pos[0]) + particle.PSFSize[0]])
        y0 = np.amax([0, int(particle.pos[1]) - particle.PSFSize[1]])
        y1 = np.amin([self.FOV[1], int(particle.pos[1]) + particle.PSFSize[1]])
        
        #Add Gaussian PSF's of particles to image:
        for i in range(x0, x1):
            for j in range(y0, y1):
                    x2 = (i - particle.pos[0])**2
                    y2 = (j - particle.pos[1])**2
                    r = 2*(x2+y2)/(particle.sigma**2)
                    self.image[i][j] += particle.intensity*np.exp(-r)
        return self.image

    @staticmethod
    def applyElectrophoreticMovement(particle, field):
        """
        The Euler integrated electric field is added to the position of particle
        particle.
        """
        particle.pos += [particle.mobility*f for f in field]

    def updateField(self, frameNumber, step):
        stepsPerPeriod = 1/(self.DELTAT*self.electricFrequency)
        phase = 2*math.pi*(math.fmod(step+frameNumber*self.stepsPerFrame,stepsPerPeriod)/stepsPerPeriod)
        self.electricField = [a*math.sin(phase) for a in self.electricAmplitude]

    def nextStep(self, frameNumber, step):
        
        #new positions for each particle:
        
        tableEntry = np.array([])
        self.updateField(frameNumber, step)
        for particle in self.particles:
            sigma = np.sqrt(2*particle.diffusionConstant)
            #random displacement dr:
            dr = np.random.normal(loc=0.0, scale=sigma, size=2)
            #move particle:
            particle.pos += dr
            self.applyElectrophoreticMovement(particle, self.electricField)
            #Apply periodic boundary conditions:
            particle.pos = (np.mod(np.subtract(particle.pos, self.MARGIN), 
                                   np.subtract(self.FOV, 2*self.MARGIN))
                            + self.MARGIN)
            tableEntry = (frameNumber, self.stepNumber, particle.particleID, particle.pos[0], particle.pos[1], self.cameraOn)
            #Add new positions to list of all positions to compare with tracked
            #tracks.
            self.tracks.append(tableEntry)
        self.stepNumber += 1
        
        
    def simulateBatch(self, maxFrames, stepsPerFrame, stepsExposed):
        
        
        #maxSteps is the number of frames the simulation will run.
        #stepsPerFrame is the number of steps in one
        #frame taken by the camera. Choose this number so it matches the FPS 
        #of the camera. stepsPerFrame = 1/(FPS*DELTAT). Choose DELTAT and FPS
        #such that stepsPerFrame is an integer. The stepsExposed is the number
        #of steps that will be accumulated in one image.
        
        #stepsExposed < stepsPerFrame
        
        try:
            assert(stepsExposed <= stepsPerFrame)
        except AssertionError as error:
            raise AssertionError(
                    "Exposure time should be less than the time per frame. \
                    stepsExposed = {}, stepsPerFrame = {}".format(
                    stepsExposed, stepsPerFrame))
        
        #make the counter boundaries global:
        self.maxFrames = maxFrames
        self.stepsPerFrame = stepsPerFrame
        self.stepsExposed = stepsExposed
        
        #number of steps the camera does not image:
        deadTime = stepsPerFrame - stepsExposed 
        
        #get time to print elapsed time:
        
        timeStarted = time = datetime.datetime.now()
        
        for frame in range(maxFrames):
            time = datetime.datetime.now() - timeStarted
            print("Time: %s Frame: %d/%d" % (time, frame, maxFrames))
            for step in range(deadTime):
                self.nextStep(frame, step)
            #redraw image. Particles only added in the loop.
            self.image = self.BACKGROUND.copy()
            self.image = self.addGaussianNoise(self.image,
                                               self.noiseBackLevel,
                                               self.gaussianNoiseLevel)
            
            self.cameraOn = True
            for step in range(stepsExposed):
                self._addStepToImage()
                self.nextStep(frame, step)
            self.frames.append(self.image)
            self.cameraOn = False

            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
