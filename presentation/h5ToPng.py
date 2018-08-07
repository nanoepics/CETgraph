# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:13:47 2018

@author: Peter
"""

import sys
from trackUtils import trackUtils
import scipy.misc

subframes = [None]
inputPath = sys.argv[1]
outputPath = sys.argv[2]

if(len(sys.argv) == 4):
   subframes[0] = int(sys.argv[3])
if(len(sys.argv) == 5):
   subframes = [int(sys.argv[3]), int(sys.argv[4])]




frames=trackUtils.loadData(inputPath, subframes=subframes)


""""  
#use this for separate images

for i, frame in enumerate(frames):
   scipy.misc.imsave(outputPath + "%d.png" % i, frame)

"""

#use this to make tracks visible in one image

sumOfFrames = frames[0]
for frame in frames[1:-1]:
   sumOfFrames += frame

scipy.misc.imsave(outputPath + ".png", sumOfFrames)



