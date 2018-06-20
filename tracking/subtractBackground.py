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


if(len(sys.argv) == 1):
   folder  = "E:\\Peter\\simulation\\runs\\29-05-18\\run2\\ParticleDiameter40_noise_10_FPS_400"
else:
   folder = sys.argv[1]
   print("Subtracting background from file at " + folder)

trackingObject = Tracking(folder, 31, 1750, 4000, 10, -1,h5name = "data.h5", FPS = -1,useFrames = -1, createTree = False)
trackingObject.currentPath = folder
maxFrames = -1
evenlySpread = True

if(maxFrames < 0 and len(trackingObject.frames) > 1000):
   print("maximum number of frames too high. Max frames set to 1000")
   maxFrames = 1000



if(evenlySpread and len(trackingObject.frames)%maxFrames != 0):
    print("Wrong maxFrames for evenly spread choise of frames.")

if(evenlySpread and maxFrames !=-1):
    frames = []
    for i in range(0,len(trackingObject.frames),int(len(trackingObject.frames)/maxFrames)):
        frames.append(trackingObject.frames[i])
    trackingObject.frames = np.array(frames)
elif(maxFrames < 0):
    print("All frames are used.")

frames = []
array = []
dimensions = list(trackingObject.frames.shape)
if(maxFrames > 0):
    dimensions[0] = np.amin([dimensions[0],maxFrames])
else:
    maxFrames = dimensions[0]

for i in range(len(trackingObject.frames)):
    array.append(trackingObject.frames[i].flatten())



print("Starting decomposition")
u, s, v = np.linalg.svd(array,full_matrices= False)
s0 = s[0]
for i in range(1,len(s)):
    s[i] = 0

print("caclulate m = u.s.v")
array = (u * s[..., None, :]) @ v

frames = []
temp = []


for i in range(dimensions[0]):
    for j in range(0,dimensions[1]*dimensions[2],dimensions[2]):
        temp.append(array[i][j:j+dimensions[2]])
    frames.append(temp)
    temp = []


background = frames[0]
frames = []
array = []

del array
del s
del u 
del v
del frames

trackingObject.saveImage(np.uint8(background), folder + "/background.png")


trackingObject.loadData()#load data again for subtraction

print("Subtracting background from frames:")

trackingObject.subtractBackground(background = np.uint16(background))

print("Saving frames:")

trackingObject.saveHDF5Data(trackingObject.frames,"frames with bg subtracted", folder + "/data_withoutBG.h5")
print("Save AVI")
trackingObject.saveAVIData(trackingObject.frames,folder + "/subtractedBG_lin")
trackingObject.saveAVIData(trackingObject.frames,folder + "/subtractedBG_log", logarithmic=True )




print('done')



