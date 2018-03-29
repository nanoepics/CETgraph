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




class Tracking:
    
    def __init__(self,folder, particleDiameter, minimumMass, maximumMass, frameMemory, micronPerPixel,createTree = True):
        
        """
        
        There are a lot of parameters. Optional parameters will clogg, so setting will be through object
        
        """
        
        self.currentPath = folder                         #path to data folder
        self.framePath = self.currentPath + "\\data.h5"   #path to hdf5 file
        self.removeBackground = False                     #subtractBG or not (for metadatafile)
        self.particleDiameter = 21                        # particle diameter in pixels
        self.minMass = minimumMass                               # minimal mass particle
        self.maxMass = maximumMass                              #maximum mass particle (intensity)
        self.maxTravelDistance  = 20                      # maximum distance travelled between frames
        self.maxFrameMemory = frameMemory                           # If particle disappears, it will be a new unique particle after this number of frames
        self.useFrames = -1                               # if < 0, all frames will be used.
        self.minimumParticleLivetime = 5                  #minimum number of frames a particle should be located. Otherwise it is deleted
        self.cameraFPS = -1                               #camera FPS if < 0, script will try to get from metadata file
        self.micronPerPixel = micronPerPixel                    #pixel size in um
        self.removeBackgroundOffset = 20                  # subtracts the image n frames before the current frame to remove bg
        self.today = datetime.datetime.now().strftime("%d-%m-%y") #get date
        self.temperature = 293                            #temp in K
        self.boltzmannConstant = 1.380649                 #(in 10^-23 JK^-1)
        self.viscosity = 0.001                            #viscosity in Pa s
        self.estimatedDiameter = 100                      # (nm) estimated particle size  (only for metadata file)
        self.runs = 0                                     #will be set depending on folder names
        self.maximumExcentricity = 0.5                    #maximum excentricity of detected particles  
        self.frames = self.loadData()                     #will be set by createDirectoryTree()
        self.frameDimensions = [0,0,0]                    #Will be set by loadData 
        self.particles   = []                             #detected particles, will be filled by detectParticles()
        
        
        if (self.cameraFPS < 0):
           exposureTime = float(self.getFromMetaData("ExposureTime", (self.currentPath +  "\\metadata.txt")))
           if(exposureTime < 0):
               print("Could not find exposure time. Set exposure time to 40 FPS")
               self.exposureTime = 1000000.0/40
           else:
               self.exposureTime = exposureTime
           self.cameraFPS = 1000000.0/self.exposureTime
        """
        create a folder for each run. If a different folder should be used, turn createTree off
        and manually call the createDirectoryTree function
        """
        if(createTree):
            self.createDirectoryTree()



    def createDirectoryTree(self, path = "", runs = 0, foldername = "", date = ""):
        if(path == ""):
           path = self.currentPath
        
        
        if (foldername == ""):
            foldername = "/tracking"
        if(date == ""):
            date = "/" + self.today
           
            
        """ I think the code in this comment is not necessary:
        for file in os.listdir(path):
           if file.endswith("data.h5"):
              print("Imported: " + file)
              self.framePath = path + "\\" + file
        """
        

            
        if not os.path.isdir(path + foldername):
            os.mkdir(path + foldername)
            
        if not os.path.isdir(path + foldername + date ):
            os.mkdir(path + foldername + date)
        
        
        files = os.listdir(path + foldername + date)
        
        #get number of runs:
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

    
    def loadData(self,dataKey = ""):

        file = h5py.File(self.framePath, 'r') # open existing file in read-only mode
        if(dataKey == ""):
            key = next(iter(file.keys())) # getting the first key
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
            
        self.frames = np.array(data)

        print("Data loaded")
        print("Pixel min/max: " + str(np.amin(data))+ "/" + str(np.amax(data)))
    
        file.close()

    
        self.maxFrames = len(self.frames)
        if(self.useFrames >= 0 and self.useFrames < self.maxFrames):
            self.maxFrames = self.useFrames
        if self.useFrames > len(self.frames):
            print("Maximum number of frames larger than number of imported frames: using number of imported frames instead. \nRequested number of frames: " + str(self.useFrames) + " Number of frames in data: " + str(len(self.frames)) )

        if(self.maxFrames < self.minimumParticleLivetime ):
            print("useFrames < minimumParticleLivetime. This means all particles will be deleted and no tracking will be done.")

        return self.frames




    def subtractBackground(self, silent = True):
        self.removeBackground = True       
        frames = [] #this will be the frames without bg
        frames0 = self.frames.astype(np.int16) # negative values are needed, negative values are set to 0 by clip
        
        if(self.subtractBackground == True):
            for i in range(len(frames0)):
                index = np.mod(i+self.removeBackgroundOffset,len(frames0))
                if ((i%int(len(frames0)/10) == 0) and silent ):
                    print(str(int(i*100/len(frames0))) + " %")
                if index < 0:
                    index = len(frames0)+index # beware '+' because index < 0
                frames.append(np.subtract(frames0[i],frames0[index]))
        else:
            frames = frames0
        
        frames = np.array(frames)
        
        np.clip(frames, 0, np.amax(frames), frames)
        frames = np.uint16(frames)
        self.frames = frames
        return frames


    def getSettings(self, dest = "", writeOutput = True):
        
        metadataText = "Run " + str(self.runs) +  ' ' + str(datetime.datetime.now()) + \
        "\n" + "Using data from file: " + self.framePath + "\n\nSettings:\n"  + \
        "particle diameter (px): " + str(self.particleDiameter) +"\n" + \
        "particle diameter (bottle): " +  str(self.estimatedDiameter) +  "\n" + \
        "minimum mass: " +  str(self.minimaleIntensiteit)+"\n" + \
        "maximum excentricity: " + str(self.maximumExcentricity) + "\n" + \
        "maximum distance between frames (px): " + str(self.maxTravelDistance)+"\n" + \
        "minimum distance between frames (px): " + str(self.minTravelDistance)+"\n" + \
        "memory (frames): " + str(self.maxFramesBeforeForgettingParticle)+"\n" + \
        "minimum number of frames of particle trajectory (frames): " + str(self.minimumParticleLivetime)+"\n" + \
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
            self.writeMetadata(metadataText, self.currentPath)
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
       self.drift = tp.compute_drift(links)
       links = tp.subtract_drift(links.copy(), self.drift)
        
       if(not silent):
           metadataText = "\n\n" + "number of detected particles: " + str(links['particle'].nunique()) + "\n" + \
           "number of filtered particles: " + str(links['particle'].nunique() -self.links['particle'].nunique()) + "\n" + \
           "number of detected and kept particles: " + str(links['particle'].nunique()) + "\n"
           self.writeMetadata(metadataText)
           
           
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
           

       if(links['particle'].nunique() == 0 ):
          print("Too few particles left")
          self.writeMetadata(metadataText  + "\n\nAll partiles filtered \n")
          
       self.links = links
          
       return links
       
       
       
       
    
    def detectParticles(self, maxFrames = 0):
        if(maxFrames == 0):
            maxFrames = self.maxFrames
        self.particles = tp.batch(self.frames[:maxFrames],self.particleDiameter,minmass=self.minMass)
        
        

    def showDetectedParticles(self, frameNumber = 0, path = "", saveImages = True, showImages = True):
        
        if(path == ""):
            path = self.currentPath
        
        particles = tp.locate(self.frames[frameNumber], self.particleDiameter, minmass=self.minMass)

        if(showImages and saveImages):
            plt.figure()
            figure = plt.gcf()
            tp.annotate(particles, self.frames[frameNumber])
            figure.savefig(path + '/detectedParticles_run' + str(self.runs) + '.png')
            plt.show(figure)
            
            fig, ax = plt.subplots()
            ax.hist(particles['mass'], bins=20)
            ax.set(xlabel='mass', ylabel='count')
            plt.savefig(path + '/massDistribution_run' + str(self.runs) + '.pdf')
            
            plt.figure()
            figure = plt.gcf()
            tp.subpx_bias(particles)
            plt.savefig(path + '/subPixelBias_run' + str(self.runs) + '.pdf')
            plt.show(figure)
        return particles
        
    
    
    
    def calculateDiffusion(self):
        self.msdData = tp.imsd(self.links, self.micronPerPixel, self.cameraFPS)
        self.fits = []
        self.diffusionConstants = []
        self.particleDiameters = []
        self.fittedPower = []
       
        for i in self.msdData.iloc[0,:].index.values:
            temp = self.msdData[i][np.isfinite(self.msdData[i]) & self.msdData[i] > 0]
            self.fits.append(tp.utils.fit_powerlaw(temp, plot=False).as_matrix().tolist() )
        for e in self.fits: 
            self.fittedPower.extend([e[0][0]])
            self.diffusionConstants.extend([4*e[0][1]])
            self.particleDiameters.extend([ (4*self.boltzmannConstant*self.temperature)/(3*math.pi*self.viscosity*100*e[0][1])])
        

    def saveAVIData(self, data, dest, format = 'XVID',colour = False):
        if "." not in dest:
            dest = dest + ".avi"
        if(np.amax(data) > 255):
            data  = (1.0*data/np.amax(data))*255.0
        data = np.array(data)
        data = data.astype(np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*format)
        videoFile = cv2.VideoWriter(dest,fourcc,self.cameraFPS,(len(data[0][0]),len(data[0])),colour)
        
        for i in range(np.shape(data)[0]):
            videoFile.write(np.uint8(data[i,:,:]))
       
        videoFile.release()
        return


    def getFromMetaData(self, text,path):
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
        if(dest == ""):
            dest = self.currentPath + "/metadata.txt"
            
        try:
            f = open(dest, "a")
            f.write(text)
            f.close()
        except:
            print("Cannot make metadata file.")
    
    
    def generateMiscPlots(self):
        plt.figure()
        figure = plt.gcf()
        figure = self.drift.plot()
        plt.savefig(self.currentPath + '/drift' + str(self.runs) + '.pdf')
        plt.show(figure)
        
        powerLaw = tp.emsd(self.links, self.micronPerPixel, self.cameraFPS)
        
        
        plt.figure()
        figure = plt.gcf()
        plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
        plt.xlabel('lag time $t$');
        trackpyFit=tp.utils.fit_powerlaw(powerLaw)
        figure.savefig(self.currentPath + '/powerLawFit' + str(self.runs) + '.pdf')
        
        n, bins, patches = plt.hist(self.diffusionConstants, 10)
        plt.xlabel('D')
        plt.ylabel('Count')
        plt.title('Histogram of Diffusion Constants')
        plt.grid(True)
        plt.show()
        
        
        n, bins, patches = plt.hist(self.particleDiameters, 10)
        plt.xlabel('a (nm)')
        plt.ylabel('Count')
        plt.title('Histogram of Diameters')
        plt.grid(True)
        plt.show()



        self.writeMetadata("\n" + str(trackpyFit))
        print(str(trackpyFit))
        
        trackpyFit = trackpyFit.as_matrix()
        
        
        diameter = (4*self.boltzmannConstant*self.temperature)/(3*math.pi*self.viscosity*100*trackpyFit[0][1])
        print("Particle diameter: " + str(diameter) + " nm" )
        
        self.writeMetadata("\nParticle diameter (nm): " + str(diameter))
        return diameter






