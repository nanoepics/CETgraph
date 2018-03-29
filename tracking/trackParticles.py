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
import trackpy as tp
import pims
import math
from PIL import Image
import os
import datetime
import h5py
import cv2
import sys #gives sys.exit() for debugging
from tracking import Tracking



folder  = "D:\\Onderzoek\\data\\19-03-2018\\run1" 
trackingObject = Tracking(folder, 21, 10000, 40000, 5, 0.225664)
folder = trackingObject.currentPath
trackingObject.saveImage(trackingObject.frames[0], folder + "/firstFrame.png")


trackingObject.subtractBackground()
trackingObject.saveAVIData(trackingObject.frames, folder + "/subtractedBackground")
trackingObject.showDetectedParticles()

trackingObject.maxFrames = 10
trackingObject.detectParticles()
trackingObject.linkParticles()
trackingObject.calculateDiffusion()
trackingObject.generateMiscPlots()





