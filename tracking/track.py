# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:32:33 2018

@author: Peter
"""

"""

This class provides code to track particles in hdf5 data format arrays



"""

import numpy as np                 #numpy arrays
import pandas                      #trackpy uses panda frames
import matplotlib.pyplot as plt    #plotting
import trackpy as tp               #trackpy
import pims                        #import output image stackss
import math                        #maths utils
from PIL import Image              # io images
import os                          #io
import datetime                    #date stamp
import h5py                        #import and export hdf5
import cv2                         # export avi
import sys                         #gives sys.exit() for debugging
import pickle
from collections import Counter    #used to count #frames in which individual particles occur: used for weighted fit


class Tracking:
    
    def __init__(self,folder, particleDiameter, minimumMass, maximumMass, frameMemory, micronPerPixel,createTree = True, dataKey = None, h5name = "data.h5",FPS = -1,useFrames = -1):
        
        """
        
        There are a lot of parameters. Optional parameters will clogg the function header, so setting will be through object
        
        """
        self.currentPath = folder                         #path to data folder
        self.framePath = self.currentPath + "\\" +h5name   #path to hdf5 file
        self.removeBackground = False                     #subtractBG or not (for metadatafile)
        self.particleDiameter = particleDiameter                        # particle diameter in pixels
        self.minMass = minimumMass                               # minimal mass particle
        self.maxMass = maximumMass                              #maximum mass particle (intensity)
        self.maxTravelDistance  = 20                      # maximum distance travelled between frames
        self.minimumMSD = 0.1                              #minimum mean square displacement for calculation D, because D is calculated for each particle individually, this parameter is not important
        self.maxFrameMemory = frameMemory                           # If particle disappears, it will be a new unique particle after this number of frames
        self.useFrames = useFrames                              # if < 0, all frames will be used.
        self.minimumParticleLivetime = 50                #minimum number of frames a particle should be located. Otherwise it is deleted
        self.cameraFPS = FPS                             #camera FPS if < 0, script will try to get from metadata file
        self.micronPerPixel = micronPerPixel                    #pixel size in um
        self.removeBackgroundOffset = 50                  # subtracts the image n frames before the current frame to remove bg
        self.today = datetime.datetime.now().strftime("%d-%m-%y") #get date
        self.temperature = 293                            #temp in K
        self.boltzmannConstant = 1.380649                 #(in 10^-23 JK^-1)
        self.viscosity = 0.001                            #viscosity in Pa s
        self.estimatedDiameter = 100                      # (nm) estimated particle size  (only for metadata file)
        self.runs = 0                                     #will be set depending on folder names
        self.maximumExcentricity = 0.5                    #maximum excentricity of detected particles  
        self.frames = self.loadData(dataKey = dataKey)                     #will be set by createDirectoryTree()
        self.frameDimensions = [0,0,0]                    #Will be set by loadData 
        self.particles   = []                             #detected particles, will be filled by detectParticles()
        self.numberOfFramesFitted = -1                     #if <0 all frames will be used.
        self.maxLagTime = 100                             #maximum number of frames used for msd calculation
        self.removeLastFrames = False #remove last frames to prevent artefacts from background removal
        
        
        if (self.cameraFPS < 0):
           exposureTime = float(self.getFromMetaData("ExposureTime", (self.currentPath +  "\\metadata.txt")))
           if(exposureTime < 0):
               print("Could not find exposure time. Set exposure time to 40 FPS")
               self.exposureTime = 1000000.0/40
           else:
               self.exposureTime = exposureTime
           self.cameraFPS = 1000000.0/self.exposureTime
        else:
            self.exposureTime = 1000000.0/self.cameraFPS
        print("FPS is: " + str(self.cameraFPS))
        """
        create a folder for each run. If a different folder should be used, turn createTree off
        and manually call the createDirectoryTree function
        """
        if(createTree):
            self.createDirectoryTree()



    def createDirectoryTree(self, path = "", runs = 0, foldername = "", date = ""):
        
        """
        This function creates directories for the date and for each run, so the data of each run is saved.
        """
        
        if(path == ""):
           path = self.currentPath
        
        
        if (foldername == ""):
            foldername = "/tracking"
        if(date == ""):
            date = "/" + self.today
        

            
        if not os.path.isdir(path + foldername):
            os.mkdir(path + foldername)
            
        if not os.path.isdir(path + foldername + date ):
            os.mkdir(path + foldername + date)
        
        
        files = os.listdir(path + foldername + date)
        
        #get number of runs from directory name:
        if(runs == 0):
            for file in files:
                if(file[3:].isdigit() and file[0:3] == 'run' and int(file[3:]) > self.runs):
                    self.runs = int(file[3:])
            self.runs  = self.runs + 1
        else:
            self.runs = runs
        
        #make the folder for the output of this particular run:
            
        self.currentPath = self.currentPath + foldername + date + '/run' + str(self.runs)
        os.mkdir(self.currentPath)
        print("Created Folder:\n" + self.currentPath  + "\n")
        return self.currentPath      

    
    def loadData(self,dataKey = None):

        file = h5py.File(self.framePath, 'r') # open existing file in read-only mode
        if(dataKey == None):
            key = next(iter(file.keys())) # getting the first key by default
        else:
            key = dataKey
        print(key + str(" dataset loaded."))
        data = file[key]# corresponding dataset
        
        self.frameDimensions = data.shape
        
        if(len(data.shape) == 3):
            print("Stack loaded. Number of frames:"  + str(data.shape[0]) + " Image: " + str(data.shape[1]) + "x" + str(data.shape[2]))
        elif(len(data.shape) == 2):
            print("Image loaded. Image: " + str(data.shape[0]) + "x" + str(data.shape[1]))
        else:
            print("Wrong number of dimensions. (Colour data?)")
            print(data.shape)
            
        #each tracking object has its own set of variables, including frames:    
        self.frames = np.array(data)

        print("Data loaded")
        print("Pixel min/max: " + str(np.amin(data))+ "/" + str(np.amax(data)))
    
        file.close()

    
        self.maxFrames = len(self.frames)
        
        #some sanity checks:
        
        if(self.useFrames >= 0 and self.useFrames < self.maxFrames):
            self.maxFrames = self.useFrames
        if self.useFrames > len(self.frames):
            print("Maximum number of frames larger than number of imported frames: using number of imported frames instead. \nRequested number of frames: " + str(self.useFrames) + " Number of frames in data: " + str(len(self.frames)) )

        if(self.maxFrames < self.minimumParticleLivetime ):
            print("useFrames < minimumParticleLivetime. This means all particles will be deleted and no tracking will be done.")

        return self.frames



    def subtractBackground(self, silent = True, background = []):
        self.removeBackground = True       
        frames = [] #this will be the frames without bg
        frames0 = self.frames.astype(np.int16) # negative values are needed, negative values are set to 0 by clip
        if(background == []):
            maxFrames = np.amin([self.maxFrames+self.removeBackgroundOffset,len(frames0)])
            for i in range(maxFrames):
                index = np.mod(i+self.removeBackgroundOffset,len(frames0))
                if ((i%int(len(frames0)/10) == 0) and not silent ):
                    print(str(int(i*100/len(frames0))) + " %")
                if index < 0:
                    index = len(frames0)+index # beware '+' because index < 0
                frames.append(np.subtract(frames0[i],frames0[index]))
            
        else:
            if(background.shape != frames0[0].shape):
                print("Wrong background shape: ")
                print(background.shape)
                print(frames0[0].shape)
            maxFrames = len(frames0)
            for i in range(maxFrames):
                frames.append(np.subtract(frames0[i],background))        


        
        frames = np.array(frames)
        #bring each pixel value below 0 back to 0:
        np.clip(frames, 0, np.amax(frames), frames)
        frames = np.uint16(frames)
        if(self.removeLastFrames):
            self.frames = frames[:-self.removeBackgroundOffset]
            self.frameDimensions = frames.shape
        else:
            self.frames = frames
        
        return frames

        
    def getSettings(self, dest = "", writeOutput = True):
        
        metadataText = "Run " + str(self.runs) +  ' ' + str(datetime.datetime.now()) + \
        "\n" + "Using data from file: " + self.framePath + "\n\nSettings:\n"  + \
        "particle diameter (px): " + str(self.particleDiameter) +"\n" + \
        "particle diameter (bottle): " +  str(self.estimatedDiameter) +  "\n" + \
        "minimum mass: " +  str(self.minMass)+"\n" + \
        "maximum mass: " +  str(self.maxMass)+"\n" + \
        "maximum excentricity: " + str(self.maximumExcentricity) + "\n" + \
        "maximum distance between frames (px): " + str(self.maxTravelDistance)+"\n" + \
        "minimum distance between frames (px): " + str(self.minimumMSD)+"\n" + \
        "memory (frames): " + str(self.maxFrameMemory)+"\n" + \
        "minimum number of frames of particle trajectory (frames): " + str(self.minimumParticleLivetime)+"\n" + \
        "max lag time ( rames): " + str(self.maxLagTime) + \
        "number of frames used (frames): " + str(self.useFrames)+"\n" + \
        "number of frames of full data (frames): " + str(self.frameDimensions[0])+"\n" + \
        "Image size: " + str(self.frameDimensions[1]) + 'x' + str(self.frameDimensions[2]) + "\n"+\
        "FPS: " + str(self.cameraFPS) +"\n" + \
        "Exposure time (us): " + str(self.exposureTime) + "\n" + \
        "Micron/pixel : " + str(self.micronPerPixel) +"\n" + \
        "Background subtracted: " + str(self.removeBackground) + "\n" 

        if(dest != "" and writeOutput):
            self.writeMetadata(metadataText, dest)
        elif(dest == "" and writeOutput):
            self.writeMetadata(metadataText, self.currentPath + "/metadata.txt")
        return metadataText
        
        

    def saveHDF5Data(self,data,key, dest):
        
        
        with h5py.File(dest, 'w') as hf:
            hf.create_dataset(key, data = data)
            hf.close()


    def saveImage(self,data, dest, bit = 16):
        
        
        if (dest[-3:] == "png" or dest[-3:] == "bmp" or bit == 8 ):
            data  = (1.0*data/np.amax(data))*255.0
            data = data.astype(np.uint8)
        figure = Image.fromarray(data)
        figure.save(dest)
        return
    
    
    def linkParticles(self, filtering = True):
       self.links  = tp.link_df(self.particles, self.maxTravelDistance , memory=self.maxFrameMemory )
       if(filtering):
           self.links = self.filterLinks()
       return self.links
    
    
    
    def filterLinks(self, silent = False):
       links = self.links
       links = tp.filter_stubs(links, self.minimumParticleLivetime)
       links = links[((links['mass'] < self.maxMass) & (links['ecc'] < self.maximumExcentricity))]
       print(links)
       self.drift = tp.compute_drift(links)
       links = tp.subtract_drift(links.copy(), self.drift)
       if(not silent):
           metadataText = "\n\n" + "number of detected particles: " + str(links['particle'].nunique()) + "\n" + \
           "number of filtered particles: " + str(links['particle'].nunique() -self.links['particle'].nunique()) + "\n" + \
           "number of detected and kept particles: " + str(links['particle'].nunique()) + "\n"
           self.writeMetadata(metadataText)
           
           """
           plot the mass versus size and plot the trajectories of filtered links:
           """ 
           plt.figure()
           figure = plt.gcf()
           tp.mass_size(links.groupby('particle').mean())
           figure.savefig(self.currentPath + '/massSizePlot' + str(self.runs) + '.pdf')
           plt.show(figure)
           
           plt.figure()
           figure = plt.gcf()
           tp.plot_traj(links);
           figure.savefig(self.currentPath + '/trajectories' + str(self.runs) + '.pdf')
           plt.show(figure)
           
       #if after filtering no particles are left, throw a warning:
       if(links['particle'].nunique() == 0 ):
          print("Too few particles left")
          self.writeMetadata(metadataText  + "\n\nAll partiles filtered \n")
          
       self.links = links
          
       return links
       
       
       
       
    
    def detectParticles(self, maxFrames = 0, silent = False):
        if(maxFrames == 0):
            maxFrames = self.maxFrames
        self.particles = tp.batch(self.frames[:maxFrames],self.particleDiameter,minmass=self.minMass)
        if(not silent):
            fig, ax = plt.subplots()
            ax.hist(self.particles['mass'], bins=20)
            ax.set(xlabel='mass', ylabel='count')
            plt.savefig(self.currentPath + '/massDistribution_run' + str(self.runs) + '.pdf')
        return

    def showDetectedParticles(self, frameNumber = 0, path = "", saveImages = True, showImages = True):
        """
        function for diagnostic purposes: it outputs an annotated image of detected particles and some 
        plots. It outputs a mass distribution of a single frame (as specified in header) and the decimal 
        value of estimated x and y positions of found particles. If This distribution is not homogeneously 
        distributed, noise is too ofted detected as particle.
        
        """ 
        if(path == ""):
            path = self.currentPath
        
        particles = tp.locate(self.frames[frameNumber], self.particleDiameter, minmass=self.minMass)

        if(showImages and saveImages):
            plt.figure()
            figure = plt.gcf()
            tp.annotate(particles, self.frames[frameNumber])
            figure.savefig(path + '/detectedParticles_run' + str(self.runs) + "_frame_" + str(frameNumber) + '.png')
            plt.show(figure)
            
            fig, ax = plt.subplots()
            ax.hist(particles['mass'], bins=20)
            ax.set(xlabel='mass', ylabel='count')
            plt.savefig(path + '/massDistribution_run' + str(self.runs) + "_frame_" + str(frameNumber) + '.pdf')
            
            plt.figure()
            figure = plt.gcf()
            tp.subpx_bias(particles)
            plt.savefig(path + '/subPixelBias_run' + str(self.runs) + "_frame_" + str(frameNumber) + '.pdf')
            plt.show(figure)
        return particles
        
    
    
    
    def calculateDiffusion(self, maxLagTime = None):
        
        """ 
        The diffusion is calculated by slicing the data in slices of length maxLagTime and fitting 
        the mean square displacement to (At)^n. n should be approximately 1 and A  = 4D.
        
        This function also stores the fitted data (fitted power, diffusion constant, particle diameter)
        per particle in different lists for future use. 
        """
        if(maxLagTime == None):
            maxLagTime = self.maxLagTime
        else:
            self.maxLagTime = maxLagTime
            
            
        self.msdData = tp.imsd(self.links, self.micronPerPixel, self.cameraFPS, max_lagtime = maxLagTime)
        self.fits = []
        self.diffusionConstants = []
        self.particleDiameters = []
        self.fittedPower = []
        self.visibleInFrames = []
        #deletes null and nan values, and minimumMSD check:
        occurance = Counter(self.links['particle'])#Count in how many frames a particle is seen.

        
        for i in self.msdData.iloc[0,:].index.values:
            temp = self.msdData[i][np.isfinite(self.msdData[i]) & self.msdData[i] > self.minimumMSD]
            if((not temp.isnull().values.any()) and (len(temp) > 0)):
                self.fits.append([i,tp.utils.fit_powerlaw(temp, plot=False).as_matrix().tolist()] )
                if((not np.isnan(self.fits[-1][1][0][0])) and (not np.isnan(self.fits[-1][1][0][1])) ):
                    self.visibleInFrames.append(occurance[i])#number of frames a particle is seen is used to give weigths to particle diamater histogram
        for e in self.fits:
            if((not np.isnan(e[1][0][0])) and (not np.isnan(e[1][0][1])) ):
                self.fittedPower.extend([e[1][0][0]])
                self.diffusionConstants.append([e[0],0.25*e[1][0][1]])#0.25 because of a == 4D
                self.particleDiameters.append([e[0],(4*self.boltzmannConstant*self.temperature)/(3*math.pi*self.viscosity*100*e[1][0][1])])
        self.diffusionConstants = np.array(self.diffusionConstants)
        self.particleDiameters = np.array(self.particleDiameters)
        

    def saveAVIData(self, data, dest, format = 'XVID',colour = False):
        #this function saves data as xvid avi format
        if "." not in dest:
            dest = dest + ".avi"
        maxPixelValue = np.amax(data)
        if(maxPixelValue > 255):
            data  = (1.0*data/maxPixelValue)*255.0
        data = np.array(data)
        data = data.astype(np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*format)
        videoFile = cv2.VideoWriter(dest,fourcc,self.cameraFPS,(len(data[0][0]),len(data[0])),colour)
        
        for i in range(np.shape(data)[0]):
            videoFile.write(np.uint8(data[i,:,:]))
       
        videoFile.release()
        return


    def getFromMetaData(self, text,path):
        """
        This function searches a metadata file generated by camera script
        in path to get stored data form that file. return -1 if failed, but file exists and -9999 if
        file does not exist:
        """
        try:
            f = open(path, "r")
            input =  f.read().split("\n")
            f.close()
            for element in input:
                element = element.split(',')
                if(len(element) < 2):
                    continue
                element[0] = element[0][2:-1]
                element[1] = element[1].replace("'", "")
                element[1] = element[1][:-1]
                if(element[0] == text):
                    return element[1]
    
            return -1
        except:
            print("Cannot find metadata file.")
            return -9999
    
    
    def writeMetadata(self, text, dest = ""):
        """
        write to text file, metadata.txt by default
        """
        if(dest == ""):
            dest = self.currentPath + "/metadata.txt"
            
        try:
            f = open(dest, "a")
            f.write(text)
            f.close()
        except:
            print("Cannot make metadata file.")
        return
    
    def generateMiscPlots(self, binsize = 20, plotMinDiameter = 0, plotMaxDiameter = 1000,maxLagTime = None):
        
        """
        This function generated various plots and writes tracking parameters to metadata.txt.
        """
        if(maxLagTime == None):
            maxLagTime = self.maxLagTime
        else:
            self.maxLagTime = maxLagTime
        
        
        plt.figure()
        figure = plt.gcf()
        figure = self.drift.plot()
        plt.savefig(self.currentPath + '/drift' + str(self.runs) + '.pdf')
        plt.show(figure)
        
        powerLaw = tp.emsd(self.links, self.micronPerPixel, self.cameraFPS,max_lagtime = maxLagTime)
        
        
        plt.figure()
        figure = plt.gcf()
        plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
        plt.xlabel('lag time $t$');
        trackpyFit=tp.utils.fit_powerlaw(powerLaw)
        figure.savefig(self.currentPath + '/powerLawFit' + str(self.runs) + '.pdf')
        
        
        n, bins, patches = plt.hist(self.diffusionConstants[:,1], binsize)
        figure = plt.gcf()
        plt.xlabel('D')
        plt.ylabel('Count')
        plt.title('Histogram of Diffusion Constants (individual partilces)')
        plt.grid(True)
        figure.savefig(self.currentPath + '/diffusionConstants' + str(self.runs) + '.pdf')
        plt.show()
        
        n, bins, patches = plt.hist(self.diffusionConstants[:,1], binsize,weights = self.visibleInFrames)
        figure = plt.gcf()
        plt.xlabel('D')
        plt.ylabel('Count')
        plt.title('Histogram of Diffusion Constants')
        plt.grid(True)
        figure.savefig(self.currentPath + '/diffusionConstantsWeighted' + str(self.runs) + '.pdf')
        plt.show()
        
        n, bins, patches = plt.hist(self.particleDiameters[:,1], binsize)
        figure = plt.gcf()
        plt.xlabel('a (nm)')
        plt.ylabel('Count')
        plt.title('Histogram of Diameters (individual partilces)')
        plt.grid(True)
        figure.savefig(self.currentPath + '/diameters' + str(self.runs) + '.pdf')
        plt.show()
        
        n, bins, patches = plt.hist(self.particleDiameters[:,1], binsize, weights = self.visibleInFrames)
        figure = plt.gcf()
        plt.xlabel('a (nm)')
        plt.ylabel('Count')
        plt.title('Histogram of Diameters')
        plt.grid(True)
        figure.savefig(self.currentPath + '/diametersWeighted' + str(self.runs) + '.pdf')
        plt.show()
        
        n, bins, patches = plt.hist(self.visibleInFrames, binsize)
        figure = plt.gcf()
        plt.xlabel('Path length in frames')
        plt.ylabel('Count')
        plt.title('Histogram of path length')
        plt.grid(True)
        figure.savefig(self.currentPath + '/pathLengthHistogram' + str(self.runs) + '.pdf')
        plt.show()
        
        n, bins, patches = plt.hist([e for e in self.particleDiameters[:,1] if ((e >= plotMinDiameter) and (e <= plotMaxDiameter))], binsize)
        figure = plt.gcf()
        plt.xlabel('a (nm)')
        plt.ylabel('Count')
        plt.title('Histogram of Diameters')
        plt.grid(True)
        figure.savefig(self.currentPath + '/diametersZoom' + str(self.runs) + '.pdf')
        plt.show()



        fig, ax = plt.subplots()
        ax.hist(self.links['mass'], bins=20)
        ax.set(xlabel='mass', ylabel='count')
        plt.grid(True)
        plt.savefig(self.currentPath + '/massDistributionFiltered_run' + str(self.runs) + '.pdf')
        plt.show()

        self.writeMetadata("\n" + str(trackpyFit))
        print(str(trackpyFit))
        
        trackpyFit = trackpyFit.as_matrix()
        
        
        diameter = (4*self.boltzmannConstant*self.temperature)/(3*math.pi*self.viscosity*100*trackpyFit[0][1])
        print("Particle diameter: " + str(diameter) + " nm" )
        self.writeMetadata("\nParticle diameter (nm): " + str(diameter))
        np.savetxt(self.currentPath +"\\diffusionConstant.csv", self.diffusionConstants, delimiter=",")
        np.savetxt(self.currentPath +"\\particleDiameters.csv", self.particleDiameters, delimiter=",")
        

        
        
        masslist = []
        #print(trackingObject.links.groupby('particle').mean())
        massDiffusion = []
        massDiameter = []
        for particleID in list(self.msdData):
            mass = self.links.loc[self.links['particle'] == particleID]['mass'].mean()
            if(particleID in self.diffusionConstants[:,0]):
                masslist.append([particleID,mass])
                indexOfID=list(self.diffusionConstants[:,0]).index(particleID)
                diffusion = self.diffusionConstants[:,1][indexOfID]
                diameter = self.particleDiameters[:,1][indexOfID]
                massDiffusion.append([mass,diffusion])
                massDiameter.append([mass,diameter])
        massDiffusion = np.array(massDiffusion)    
        massDiameter = np.array(massDiameter)
        
        plt.scatter(massDiffusion[:,0],massDiffusion[:,1])
        figure = plt.gcf()
        plt.xlabel("Mass")
        plt.ylabel("Diffusion Constant")
        plt.title("Diffusion versus Mass")
        plt.grid(True)
        figure.savefig(self.currentPath + '/MassDiffusionScatterplot' + str(self.runs) + '.pdf')    
        plt.show()
        
        plt.scatter(massDiameter[:,0],massDiameter[:,1])
        figure = plt.gcf()
        plt.xlabel("Mass")
        plt.ylabel("Particle Diameter (nm)")
        plt.title("Diameter versus Mass")
        plt.grid(True)
        figure.savefig(self.currentPath + '/MassDiameterScatterplot' + str(self.runs) + '.pdf')    
        plt.show()
        
        f = open(self.currentPath + "\\trackingPickleObject.pyc", "wb")
        print("Dataframes deleted to pickle object")
        self.frames = []
        pickle.dump(self,f)
        f.close()
        
        return diameter






