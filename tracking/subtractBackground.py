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
from trackUtils import trackUtils

maxFrames = 50
evenlySpread = True

if(len(sys.argv) == 1):
   folder  = "D:\\Onderzoek\\data\\18-07-27\\18-07-27\\run3"
else:
   folder = sys.argv[1]
   print("Subtracting background from file at " + folder)


if(folder[-3] == ".h5"):
   file = folder
else:
   file = folder + "\\data.h5"
print(file)

frames = trackUtils.loadData(file)


if(maxFrames < 0 and len(frames) > 25):
   print("maximum number of frames too high. Max frames set to 500")
   maxFrames = 25


if(evenlySpread and len(frames)%maxFrames != 0):
    print("Wrong maxFrames for evenly spread choise of frames.")

if(evenlySpread and maxFrames != -1):
    tempframes = []
    for i in range(0,len(frames),int(len(frames)/maxFrames)):
        tempframes.append(frames[i])
    frames = np.array(tempframes)
elif(maxFrames < 0):
    print("All frames are used.")

tempframes = []
array = []
dimensions = list(frames.shape)
if(maxFrames > 0):
    dimensions[0] = np.amin([dimensions[0],maxFrames])
else:
    maxFrames = dimensions[0]

for i in range(len(frames)):
    array.append(frames[i].flatten())



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

trackUtils.saveImage(np.uint8(background), folder + "/background.png")


frames = trackUtils.loadData(file)#load data again for subtraction

print("Subtracting background from frames:")


frames = trackUtils.subtractBackground(frames, background = np.uint16(background))

print("Saving frames:")

trackUtils.saveHDF5Data(frames,"frames with bg subtracted", folder + "/data_withoutBG.h5")
print("Save AVI")
trackUtils.saveAVIData(frames,folder + "/subtractedBG_lin")
trackUtils.saveAVIData(frames,folder + "/subtractedBG_log", logarithmic=True )




print('done')



