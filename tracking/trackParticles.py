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







folder  = "D:\\Onderzoek\\data\\11-04-2018\\run10"
folder = "D:\\Onderzoek\\python\\simulation\\runs\\18-04-18\\run1\\ParticleDiameter100_noise_10_FPS_40"

"""
create a tracking object.
tracking(path to data, particle diameter (px), minmass, maxmass, 
min number of frames for detected particle,micron per pixel)
"""
#trackingObject = Tracking(folder, 31, 2000, 4000, 10, 0.225664, h5name = "simulated data.h5")
trackingObject = Tracking(folder, 11, 500, 10000, 10, 0.225664, h5name = "data.h5", FPS = 40,useFrames = 15)
folder = trackingObject.currentPath
trackingObject.saveImage(trackingObject.frames[0], folder + "/firstFrameRaw.png")



#trackingObject.subtractBackground()

trackingObject.saveImage(trackingObject.frames[0], folder + "/firstFrameWithoutBG.png")

#trackingObject.saveAVIData(trackingObject.frames, folder + "/subtractedBackground")
trackingObject.showDetectedParticles()

trackingObject.minimumMSD = 0.5 #minimum mean square displacement. Use to prevent stuck particles.
trackingObject.detectParticles() #diagnostic function. plot detected particles of first frame
trackingObject.linkParticles() #link different frames


trackingObject.calculateDiffusion(maxLagTime = 5) #maxLagTime is how many frames will be used per fit
trackingObject.generateMiscPlots(binsize = 25) #generate histograms of data
trackingObject.getSettings()   #write to metadata


