# -*- coding: utf-8 -*-saveAVIData(self, data, dest, format = 'XVID',colour = False):
"""
Created on Thu Mar 29 15:27:11 2018

@author: Peter
"""

"""
This script uses the tracking class to obtain particle data from imaged frames
"""



import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas
import trackpy as tp
import pims
import math
from PIL import Image
import os
import datetime
import h5py
import cv2
import pickle
import sys #gives sys.exit() for debugging
from track import Tracking



plt.ioff()
if(len(sys.argv)<2):
   folder = "E:\\Peter\\simulation\\runs\\29-05-18\\run2\\ParticleDiameter40_noise_10_FPS_400"
else:
   folder = sys.argv[1]

if(folder[-3:] == ".h5"):
   print(folder)
   h5name = folder.split("\\")[-1]
   a = folder.split("\\")[:-1]
   b = ["\\"]*len(a)
   list = np.array([[a[i],b[i]] for i in range(len(a))]).flatten()
   print(list)
   folder = "".join([i for i in list])
   print(folder)
else:
   h5name = "data_withoutBG.h5"




"""
create a tracking object.
tracking(path to data, particle diameter (px), minmass, maxmass, 
min number of frames for detected particle,micron per pixel)
"""

if(len(sys.argv) > 2):
   beginFrame=int(sys.argv[2])
   endFrame = int(sys.argv[3])
   numberOfFrames = endFrame-beginFrame
   if(beginFrame < 0 or endFrame < 0):
      subframes = [None, None]
      numberOfFrames = -1
   else:
      subframes = [beginFrame, endFrame]
else:
   subframes = [None, None]
   numberOfFrames = -1

trackingObject = Tracking(folder, -1, -1,-1,3, -1, h5name = h5name, FPS = -1 ,useFrames = -1, subframes=subframes,createTree = False)
if(numberOfFrames < 0):
   numberOfFrames = len(trackingObject.frames)
   beginFrame = 0
   endFrame = numberOfFrames

trackingObject.createDirectoryTree(path = folder)
folder = trackingObject.currentPath

#trackingObject.saveImage(trackingObject.frames[0], folder + "/firstFrameRaw.png")



#trackingObject.subtractBackground()

#trackingObject.saveImage(trackingObject.frames[0], folder + "/firstFrameWithoutBG.png")

#trackingObject.saveAVIData(trackingObject.frames, folder + "/subtractedBackground")

#trackingObject.showDetectedParticles() #diagnostic function. plot detected particles of first frame



trackingObject.minimumMSD = 0.5 #minimum mean square displacement. Use to prevent stuck particles.
trackingObject.detectParticles(silent = False) 
trackingObject.linkParticles( silent = False) #link different frames


trackingObject.calculateDiffusion(maxLagTime = 5) #maxLagTime is how many frames will be used per fit
trackingObject.calculateMobility(direction = 'y')
trackingObject.generateMiscPlots(binsize = 25, silent = False) #generate histograms of data
trackingObject.getSettings()   #write to metadata
trackingObject.writeMetadata("\nFrames: " + str(beginFrame) + " to " + str(endFrame) + "\n", trackingObject.currentPath + "/metadata.txt")

command =  "E:\\Peter\\Anaconda\\python.exe"  + " " + '"E:\\Peter\\python scripts\\plotHistogramSilent.py"' + " " + trackingObject.currentPath
os.system(command)



