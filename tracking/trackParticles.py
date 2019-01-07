"""
Created on Thu Mar 29 15:27:11 2018

@author: Peter
"""

"""
This script uses the tracking class and trackUtils class to obtain particle
tracks from imaged data. It can be called from the terminal with:
python trackParticles.py [path to data] [first frame] [last frame]

[first frame] and [last frame] are optional.
                                                          
"""

import matplotlib as mpl
mpl.use('Agg') # this prevents popups when processing data.
import matplotlib.pyplot as plt# plotting
plt.ioff()
import numpy as np
import os
import pickle
import sys
from track import Tracking
from trackUtils import trackUtils


if (len(sys.argv) < 2):#default folfer for debugging. Remove this before uploading.
    folder = "D:\\Onderzoek\\data\\18-08-02\\run30"
else:
    folder = sys.argv[1]


"""
If the folder is given, the following code looks for data_withoutBG.h5. If
a path to an h5 file is given, it splits the folder name from the path.
"""

if (folder[-3:] == ".h5"):
    print(folder)
    h5name = folder.split("\\")[-1]
    a = folder.split("\\")[:-1]
    b = ["\\"] * len(a)
    list = np.array([[a[i], b[i]] for i in range(len(a))]).flatten()[:-1]
    print(list)
    folder = "".join([i for i in list])
    print(folder)
else:
    h5name = "withoutBGUsingAllMeasurements.h5"

"""
The number of subframes can be given as an argument to this script. This will
be stored in the variable subframes.
"""

if (len(sys.argv) > 2):
    beginFrame = int(sys.argv[2])
    endFrame = int(sys.argv[3])
    numberOfFrames = endFrame - beginFrame
    if (beginFrame < 0 or endFrame < 0):
        subframes = [None, None]
        numberOfFrames = -1
    else:
        subframes = [beginFrame, endFrame]
else:
    subframes = [None, None]
    numberOfFrames = -1



"""
create a tracking object.
tracking(path to data, particle diameter (px), minmass, maxmass, 
min number of frames for detected particle,micron per pixel)
Optional parameters are:
h5name the name of the h5file, FPS, the frame per seconds. If this is smaller
than 0, load from a metadata file. useFrames is can be set to use only the first
few frames. This argument is superseded by subframes. subframes is an argument
that is an array with which a window of frames can be selected.
subframes = [first frame, last frame]. If subframes is [None,None], all frames
will be used.
"""

trackingObject = Tracking(folder, -1, -1, -1, 3, -1, h5name=h5name, FPS=-1, 
                          useFrames=-1, subframes=subframes)

if (numberOfFrames < 0):
    numberOfFrames = len(trackingObject.frames)
    beginFrame = 0
    endFrame = numberOfFrames

rus, outputPath = trackUtils.createDirectoryTree(folder)
trackingObject.currentPath = outputPath

trackingObject.minimumMSD = 0.05  # minimum mean square displacement. Use to prevent stuck particles.
trackingObject.detectParticles(silent=False)
trackingObject.linkParticles(silent=False, filtering=False)  # link different frames

trackingObject.calculateDiffusion(maxLagTime=5)  # maxLagTime is how many frames will be used per fit
trackingObject.calculateMobility(direction='y')
trackUtils.writeOutputToFolder(outputPath, trackingObject, metadataFile="metadata.txt")
trackUtils.generateMiscPlots(trackingObject, binsize=25, silent=False)  # generate histograms of data


f = open(outputPath + "\\trackingPickleObject.pyc", "wb")
print("Dataframes deleted to pickle object")
trackingObject.frames = []
pickle.dump(trackingObject, f)
f.close()

command = "E:\\Peter\\Anaconda\\python.exe" + " " + '"E:\\Peter\\python scripts\\plotHistogramSilent.py"' + " " + trackingObject.currentPath
os.system(command)


