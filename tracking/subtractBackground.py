# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:34:07 2018

@author: Peter
"""

import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp
import math
from PIL import Image
import os
import datetime
import h5py
import cv2
import pickle
import csv
import sys #gives sys.exit() for debugging
from track import Tracking


folder = "D:\\Onderzoek\\python\\simulation\\runs\\17-04-18\\run6\\ParticleDiameter100_noise_10_FPS_40"
folder  = "D:\\Onderzoek\\data\\11-04-2018\\run10" 


trackingObject = Tracking(folder, 31, 1750, 4000, 10, 0.225664, h5name = "data.h5", FPS = 40,useFrames = -1,makeTree = False)
trackingObject.currentPath = folder
maxFrames = 50

array = []
dimensions = list(trackingObject.frames.shape)
if(maxFrames > 0):
    dimensions[0] = np.amin([dimensions[0],maxFrames])
else:
    maxFrames = dimensions[0]



for i in range(np.amin([len(trackingObject.frames),maxFrames])):
    array.append(trackingObject.frames[i].flatten())



u, s, v = np.linalg.svd(array,full_matrices= False)
s0 = s[0]
for i in range(1,len(s)):
    s[i] = 0


array = (u * s[..., None, :]) @ v

frames = []
temp = []


for i in range(dimensions[0]):
    for j in range(0,dimensions[1]*dimensions[2],dimensions[2]):
        temp.append(array[i][j:j+dimensions[2]])
    frames.append(temp)
    temp = []


background = frames[0]


trackingObject.saveImage(np.uint8(background), folder + "/background.png")

trackingObject.subtractBackground(background = np.uint16(background))


trackingObject.saveAVIData(trackingObject.frames,folder + "/subtractedBG")

trackingObject.saveHDF5Data(trackingObject.frames,"frames with bg subtracted", folder + "/data_withoutBG.h5")



print('done')



