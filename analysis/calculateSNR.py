
# coding: utf-8

# In[110]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import math
import statistics
import pandas
from PIL import Image, ImageFilter
import importlib.util
import scipy.optimize


spec = importlib.util.spec_from_file_location("trackUtils.trackUtils", "D:\\Onderzoek\\git\\tracking\\trackUtils.py")
trackUtils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trackUtils)

#path = "D:\\Onderzoek\\data\\6-6\\shortlist\\run3\\data.h5"
path = "D:\\Onderzoek\\data\\18-08-08\\20 Hz\\run1\\data.h5"
pathToTracks = "D:\\Onderzoek\\data\\18-08-08\\20 Hz\\run1\\tracking\\18-09-21\\run3\\tracks0.csv"

path = "D:\\Onderzoek\\data\\simulation\\18-09-14\\data.h5"
pathToTracks = "D:\\Onderzoek\\data\\simulation\\18-09-14\\tracking\\18-09-25\\run5\\tracks0.csv"

gaussianBlurRadius = 5
kernelDiskWidth = 10 # same as tracking diameter.
binaryThreshold = 20


# In[111]:


def makeDiskShapedKernel(width):
    radius = 0.5*width
    radius2 = (radius*radius-0.25)#0.25 to put the entire pixel into the circle, instead of only the middle
    kernel = np.zeros((width,width))
    for i in range(width):
        for j in range(width):
            r2 = (i-radius+0.5)**2+(j-radius+0.5)**2
            if(r2 < radius2):
                kernel[i][j] = 1
    return kernel


# In[112]:


def getParticleSize(frame, x, y, radius):
    x_d = int(np.round(x))
    y_d = int(np.round(y))
    xmin = np.amax([x_d-math.ceil(radius/2), 0])
    xmax = np.amin([x_d+math.ceil(radius/2), len(frame[0])])
    ymin = np.amax([y_d-math.ceil(radius/2), 0])
    ymax = np.amin([y_d+math.ceil(radius/2), len(frame)])

    xprofile = frame[y_d, xmin:xmax]
    yprofile = frame[ymin:ymax, x_d]
    
        
    fitParametersX, varianceMatrixX = scipy.optimize.curve_fit(gauss, range(len(xprofile)), xprofile, p0 = [1000,10, 10], bounds = ([0,0,0],[5000,20,np.inf]))
    fitParametersY, varianceMatrixY = scipy.optimize.curve_fit(gauss, range(len(yprofile)), yprofile, p0 = [1000,10, 10], bounds = ([0,0,0],[5000,20,np.inf]))

    return np.mean([fitParametersX[2], fitParametersY[2]])
    
    
    
    
    
        


# In[113]:


def gauss(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/(2.0*sigma**2))

    


# In[114]:


frames = trackUtils.trackUtils.loadData(path)
blurredFrames = [cv2.GaussianBlur(frame, (gaussianBlurRadius, gaussianBlurRadius), 0) for frame in frames]
kernel = makeDiskShapedKernel(kernelDiskWidth+1)
dilationKernel = makeDiskShapedKernel(kernelDiskWidth)
backgroundFrames = [cv2.filter2D(frame, -1, kernel/np.sum(kernel)) for frame in frames]
masks = []


# In[115]:


binary = []
for i, frame in enumerate(blurredFrames):
    mask = np.subtract(frame, backgroundFrames[i], dtype=np.int32)
    mask = np.clip(mask, binaryThreshold, binaryThreshold+1)-binaryThreshold
    binary.append(mask)
    mask = cv2.dilate(1.0*mask, np.uint8(dilationKernel), iterations = 1)
    mask = 1 - mask
    masks.append(mask)


# In[116]:


pixels = []
for i, frame in enumerate(frames):
    pixels.extend(frame[masks[i].astype(bool)])
pixels = list(pixels)
print(len(pixels))
print(np.array(pixels).shape)
pixels = [float(p) for p in pixels]
meanBackground = np.mean(pixels)
variance = statistics.variance(pixels)
noise = np.sqrt(variance)

plt.figure()
plt.hist(pixels, 20)
plt.show()

print("sum: %d. avg: %f. frac px used: %f. Variance (sigma^2): %f, strd dev (sigma): %f" % (np.sum(pixels), meanBackground, len(pixels)/(len(frames)*len(frames[0])*len(frames[0][0])), variance, noise))

    


# In[117]:


def getMaxPixelInDisk(frame, x, y, width):
    radius = 0.5*width
    x_d = int(np.round(x))
    y_d = int(np.round(y))
    maxPixelValue = 0
    radius2 = (radius*radius-0.25)#0.25 to put the entire pixel into the circle, instead of only the middle
    xmin = np.amax([x_d-math.ceil(radius/2), 0])
    xmax = np.amin([x_d+math.ceil(radius/2), len(frame[0])])
    ymin = np.amax([y_d-math.ceil(radius/2), 0])
    ymax = np.amin([y_d+math.ceil(radius/2), len(frame)])
    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            r2 = (i-y_d+0.5)**2+(j-x_d+0.5)**2
            if(r2 < radius2):
                if(frame[i][j] > maxPixelValue):
                    maxPixelValue = frame[i][j]
    return maxPixelValue
    
    


# In[118]:


links = pandas.read_csv(pathToTracks)
signalList = []
sigmaList = []

for index in links.index.values:
    row = links.iloc[index]
    signalList.append(getMaxPixelInDisk(frames[int(row['frame'])], row['x'], row['y'], kernelDiskWidth))
    sigmaList.append(getParticleSize(frames[int(row['frame'])], row['x'], row['y'], 2*kernelDiskWidth))
rawSignal = np.mean(signalList)
signal = (rawSignal- meanBackground)
particleSigma = np.mean(sigmaList)
SNR = signal/noise
print("SNR = %f, signal (BG corrected) = %f signal = %f, noise = %f, background = %f, particle size = %f (px)" % (SNR, signal, rawSignal,noise, meanBackground, particleSigma))


# In[119]:


plt.figure()
plt.imshow(frames[0])
plt.show()

plt.figure()
plt.imshow(blurredFrames[0])
plt.show()

plt.figure()
plt.imshow(backgroundFrames[0])
plt.show()

plt.figure()
plt.imshow(binary[0])
plt.show()


plt.figure()
plt.imshow(masks[0])
plt.show()



print("(%d %d), (%d, %d), (%d, %d) , (%f, %f)" % (np.amin(frames[0]),np.amax(frames[0]),
                                       np.amin(blurredFrames[0]), np.amax(blurredFrames[0]),
                                       np.amin(backgroundFrames[0]), np.amax(backgroundFrames[0]),
                                       np.amin(masks[0]), np.amax(masks[0])
                                      ))


# In[120]:


getMaxPixelInDisk(frames[0], 100, 220, 20)
getMaxPixelInDisk(frames[0], 220, 100, 20)


# In[121]:


getParticleSize(frames[0], 240, 227, 20)

