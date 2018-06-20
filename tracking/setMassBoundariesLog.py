# -*- coding: utf-8 -*-saveAVIData(self, data, dest, format = 'XVID',colour = False):
"""
Created on Thu Mar 29 15:27:11 2018

@author: Peter
"""

"""
This script uses the tracking class to obtain particle data from imaged frames
"""



import numpy as np
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

particleDiameter = 15
minmass = 1000
maxmass = 10000

trackingObject = Tracking(folder, particleDiameter,minmass, maxmass,3, -1, h5name = "data_withoutBG.h5", FPS = -1 ,useFrames = -1, subframes=subframes,createTree = False)

trackingObject.frames = trackingObject.frames[0:5]

while (True):
   try:
      minmass = int(input("minimum mass: "))
      maxmass = int(input("maximum mass:"))
      particleDiameter = int(input("Particle diameter: "))
   except:
      break
   text = "Minimum mass, " + str(minmass) + "\n" + \
   "Maximum mass, " + str(maxmass) + "\n" + \
   "Diameter, " + str(particleDiameter) + "\n"

   trackingObject.minMass = minmass
   trackingObject.maxMass = maxmass
   trackingObject.particleDiameter = particleDiameter
   
   if(numberOfFrames < 0):
      numberOfFrames = len(trackingObject.frames)
      beginFrame = 0
      endFrame = numberOfFrames

   trackingObject.showDetectedParticles(saveImages = False, annotatedFrame = True,invert=True,logarithm=True)

   

dest = folder+ "\\instructions.txt"
print("Tracking instructions will be written to: " + dest)
try:
   f = open(dest, "w")
   f.write(text)
   f.close()
except:
   print("Cannot make instructions file.")




