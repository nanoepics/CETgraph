
# coding: utf-8

# In[53]:


import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import cv2
import math
import statistics
import pandas
from PIL import Image, ImageFilter
import importlib.util
import scipy.optimize
import trackpy as tp



import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("trackUtils.trackUtils", "D:\\Onderzoek\\git\\tracking\\trackUtils.py")
trackUtils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trackUtils)


if(len(sys.argv) < 2):
    path = "D:\\Onderzoek\\data\\18-08-08\\6 Hz\\run1\\exposedImage.png"
else:
    path = sys.argv[1]

splittedPath = path.split('\\')
folderPath = ""
print(splittedPath[-1][-4:])
if(splittedPath[-1][-4:] == ".png"):
    for e in splittedPath[:-1]:
        folderPath += (e + "\\")
else:
    folderPath = path
    path += "exposedImage.png"

print("saving to %s" % folderPath)
    
try:
    image = Image.open(path)
except:
    print ("Unable to load image")
    



gaussianBlurRadius = 5 #According Savin and Doyle this should be 1, but sometimes noisier data requires larger radius
kernelDiskWidth = 15 #Larger than apparent particle size in px. This is used to select canditate traces
kernelDiskParticleTrace = 5 # same as tracking diameter. This is smaller than kernelDiskWidth. 

binaryThreshold = 20 #arbitrary, depends on acquisition, so different each experiment

selectPixelsWithValue = 0.0 
minArea = 40 #minimum area of cropped image
borderMargin = 10 # does not select particles too close to the boundary 
minMass = 350 #minimum intenstiy required to suppress local maxima in noise outside tracks. Inside tracks are supressed by selecting largest distance
kernelDiskParticleTrace = 20 #dilation kernel size to find local maxima for particle trace sizes.




# In[54]:


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


def getMassInDisk(frame, x0, y0, radius):
    sum = 0
    for i in range(-radius, radius +1):
        for j in range(-radius, radius +1):
            if(i*i+j*j <= radius**2):
                x = x0 + i
                y = y0 + j
                if( x < 0 or y < 0 or x >= len(frame) or y >= len(frame[0])):
                    continue
                sum += frame[x][y]
    return sum

def selectSameValues(image, value, pixelsInArea, i0, j0):    
    imin, jmin, imax, jmax = (0,0,len(image), len(image[0]))
    for i in range(-1,2):
        for j in range(-1, 2):
            if(i == 0 and j == 0):
                continue
            i1 = i + i0
            j1 = j + j0
            if(i1 < imin or i1 >= imax or j1 < jmin or j1 >= jmax):
                continue
            if(image[i1][j1] == value):
                pixelsInArea.append([i1, j1])
                image[i1][j1] = value +1
                selectSameValues(image, value, pixelsInArea, i1, j1)
                
def gauss(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/(2.0*sigma**2))


# In[55]:



blurredImage = cv2.GaussianBlur(np.array(image), (gaussianBlurRadius, gaussianBlurRadius), 0)
kernel = makeDiskShapedKernel(kernelDiskWidth+1)
dilationKernel = makeDiskShapedKernel(kernelDiskWidth)
backgroundImage = cv2.filter2D(np.array(image), -1, kernel/np.sum(kernel))
masks = []


# In[56]:


binary = []

mask = np.subtract(image, backgroundImage, dtype=np.int32)
mask = np.clip(mask, binaryThreshold, binaryThreshold+1)-binaryThreshold
binary = mask
mask = cv2.dilate(1.0*mask, np.uint8(dilationKernel), iterations = 1)
mask = 1 - mask


# In[57]:


pixels = []

pixels.extend(np.array(image)[mask.astype(bool)])
#pixels = list(pixels)
#print(len(pixels))
#print(np.array(pixels).shape)
pixels = [float(p) for p in pixels]
meanBackground = np.mean(pixels)
variance = statistics.variance(pixels)
noise = np.sqrt(variance)

#plt.figure()
#plt.hist(pixels, 20)
#plt.show()
print("Properties of background:")
print("sum: %d. avg: %f. frac px background: %f. Variance (sigma^2): %f, strd dev (sigma): %f" % (np.sum(pixels), meanBackground, len(pixels)/(len(np.array(image))*len(np.array(image)[0])), variance, noise))

    


# In[58]:


plt.figure()
plt.imshow(image)
plt.show()

plt.figure()
plt.imshow(blurredImage)
plt.show()

plt.figure()
plt.imshow(backgroundImage)
plt.show()

plt.figure()
plt.imshow(binary)
plt.show()


plt.figure()
plt.imshow(mask)
plt.show()



print("(%d %d), (%d, %d), (%d, %d) , (%f, %f)" % (np.amin(image),np.amax(image),
                                       np.amin(blurredImage), np.amax(blurredImage),
                                       np.amin(backgroundImage), np.amax(backgroundImage),
                                       np.amin(mask), np.amax(mask)
                                      ))


# In[ ]:



                


# In[72]:




imagesOfTraces = []
copyOfMask = mask.copy()
pixelsInArea = []



for i in range(len(mask)):
    for j in range(len(mask[i])):
        if(copyOfMask[i][j] == selectPixelsWithValue):
            pixelsInArea.append([i, j])
            selectSameValues(copyOfMask, selectPixelsWithValue, pixelsInArea,i, j)
            if(len(pixelsInArea) < minArea):
                continue
            p0 = [np.amin(np.array(pixelsInArea)[:,0]), np.amin(np.array(pixelsInArea)[:,1])]
            p1 = [np.amax(np.array(pixelsInArea)[:,0]), np.amax(np.array(pixelsInArea)[:,1])]
            
            pixelsInArea = []
            
            
            if(p0[0]<borderMargin or p0[1] < borderMargin or p1[0] > (len(copyOfMask) - borderMargin) or p1[1] > (len(mask[0])-borderMargin)):
                continue
            
            trace = np.array(image)[p0[0]:p1[0], p0[1]:p1[1]].astype(np.int32())
            trace = trace - 4095*np.array(mask)[p0[0]:p1[0], p0[1]:p1[1]].astype(np.uint16())
            trace = np.clip(trace, 0, 65535).astype(np.uint16)
            imagesOfTraces.append(trace)

#plt.figure()
#plt.imshow(copyOfMask)
#plt.show()

del(copyOfMask)
len(imagesOfTraces)
            


# In[78]:



dilationKernel = makeDiskShapedKernel(kernelDiskParticleTrace+1)

distances = []
angles = []
detectedPoints = []


try:
    os.mkdir(folderPath + "detectedTraces")
except:
    print("Could not create folder in %s" % (folderPath + "detectedTraces"))

for traceNumber, traceImage in enumerate(imagesOfTraces):

    dilatedImage  = cv2.dilate(np.array(traceImage), np.uint8(dilationKernel), iterations = 1).astype(np.uint16())
    plt.figure()
    plt.imshow(imagesOfTraces[0])
    plt.show()

    plt.figure()
    plt.imshow(dilatedImage)
    plt.show()

    selectedPixels = (traceImage + 1) - dilatedImage

    plt.figure()
    plt.imshow(selectedPixels)
    plt.show()



    
    candidatePixels = []
    for i in range(len(selectedPixels)):
        for j, e in enumerate(selectedPixels[i]):
            if(e != 1):
                continue
            if(getMassInDisk(traceImage, i, j, int(kernelDiskParticleTrace/2)) > minMass):
                candidatePixels.append([i,j])

    maxDistance = 0
    angle = 0
    points = [[0,0],[0,0]]
    
    
    
    for i, e1 in enumerate(candidatePixels):
        for j, e2 in enumerate(candidatePixels):
            distance2 = (e1[0]-e2[0])**2+(e1[1]-e2[1])**2
            if(distance2 > maxDistance):
                maxDistance = distance2
                points = [e1, e2]
                
                if((e1[0]-e2[0]) != 0):
                    angle = math.atan((e1[1]-e2[1])/(e1[0]-e2[0]))
                    if(angle < 0):
                        angle = math.pi+angle
                else:
                    angle = 0.5*math.pi
                    
                    
    maxDistance = np.sqrt(maxDistance)
    
    
    distances.append([traceNumber, maxDistance])
    angles.append([traceNumber, angle])
    detectedPoints.append( [traceNumber, points[0][0], points[0][1], points[1][0], points[1][1]])
    
    
    exportImage = Image.fromarray(traceImage.astype(np.int32))
    exportImage.save(folderPath + "detectedTraces\\trace%d_16bit.png" % (traceNumber))
    exportImage = Image.fromarray(traceImage.astype(np.uint8))
    exportImage.save(folderPath + "detectedTraces\\trace%d_8bit.png" % (traceNumber))
    
    plt.figure()
    plt.imshow(traceImage)
    plt.plot([points[0][1], points[1][1]],[points[0][0], points[1][0]],"-wx")
    plt.savefig(folderPath + "detectedTraces\\processedTrace%d.png" % (traceNumber))
    plt.show()


np.savetxt(folderPath + "detectedTraces\\lengths.csv", distances, delimiter=",")
np.savetxt(folderPath + "detectedTraces\\angles.csv", angles, delimiter=",")
np.savetxt(folderPath + "detectedTraces\\points.csv", detectedPoints, delimiter=",")
           


# In[61]:


detectedPoints


# In[62]:


a = [2].extend(points)


# In[34]:


a


# In[76]:


trace

