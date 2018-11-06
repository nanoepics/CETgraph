"""
   CETgraph.tracking.autoCropper.py
   ==================================
    This script uses methods similar to [1] and [2] to features monochromatic
    microscope images. This should be used together with subtractBackground.py.
    It outputs cropped images and a crude estimate of the length. For precise
    length of the features a specific script for the experiment should be used
    on the output data from this script.
   
   
   [1] T Savin and P. Doyle, Static and Dynamic Errors in Paticle Tracking
   Microrheology, Biophysical Journal, 88, 623-638, 2005.
   
   [2] J.C. Crocker and D. G. Grier, Methods of Digital Video Microscopy for 
   Colloidal Studies
, Journal of Colloid and Interface Science, 179, 298-310, 1996

    .. lastedit:: 06-11-2018
    .. sectionauthor:: Peter Speets <p.n.a.speets@students.uu.nl>
"""

import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg') #Suppresses image display
import cv2
import math
import statistics
from PIL import Image, ImageFilter
import importlib.util
import scipy.optimize
import matplotlib.pyplot as plt



#load trackutils.
spec = importlib.util.spec_from_file_location(
        "trackUtils.trackUtils", "D:\\Onderzoek\\git\\tracking\\trackUtils.py")
trackUtils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trackUtils)

#Found features that are smaller or larger than these dimensions will be
#rejected. Numbers in pixels.

minWidth = 90
maxWidth = 200
minHeight = 45
maxHeight = 100
minArea = 100 

#According Savin and Doyle the blur radius should be 1,
#but sometimes noisier data requires larger radius.

gaussianBlurRadius = 7 

 #The kernel disk width is larger than apparent particle size in px. 
 #This is used to select canditate features.
kernelDiskWidth = 40

# same as tracking diameter. This is smaller than kernelDiskWidth. 
kernelDiskParticleTrace = 20 

#arbitrary, depends on acquisition, so different each experiment. Higher
#setting means that less traces are found.
binaryThreshold = 10 

# Do not select particles too close to the boundary. All features within this
#distance in px are rejected.
borderMargin = 10 

#minimum intenstiy required to suppress local maxima in noise outside tracks.
#Inside tracks are supressed by selecting largest distance. If the feature 
#selection parameters are chosen well, the closing and minArea should make
#this condition redundant.
minMass = 350 

#The script always assumes bright features and dark background, so this should
#be 0.0.
selectPixelsWithValue = 0.0 



#loading data if this script is run from IDE instead of terminal.
if(len(sys.argv) == 1):
    path = "D:\\Onderzoek\\data\\Tang\\image4.png"
    
else:
    path = sys.argv[1]


#Find the path of the input image:

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
    if((np.array(image).shape)[2]==3):
        image = np.array(image)[:,:,0]
except:
    print ("Unable to load image")
    



def makeDiskShapedKernel(width):
    """
    This function outputs a square array with a disk with width and height
    the same as width 
    """
    radius = 0.5*width
    #0.25 subtracted from radius**2 to put the entire pixel into the circle,
    #instead of only the middle.
    radius2 = (radius*radius-0.25)
    kernel = np.zeros((width,width))
    for i in range(width):
        for j in range(width):
            r2 = (i-radius+0.5)**2+(j-radius+0.5)**2
            if(r2 < radius2):
                kernel[i][j] = 1
    return kernel



def getParticleSize(frame, x, y, radius):
    """
    This function accepts a frame, position and a radius and performs a 
    Gaussian fit to provide an estimate of the radius in pixels. This value
    can be used estimate the apparent disk size as conditions for this script.
    """
    x_d = int(np.round(x))
    y_d = int(np.round(y))
    xmin = np.amax([x_d-math.ceil(radius/2), 0])
    xmax = np.amin([x_d+math.ceil(radius/2), len(frame[0])])
    ymin = np.amax([y_d-math.ceil(radius/2), 0])
    ymax = np.amin([y_d+math.ceil(radius/2), len(frame)])

    xprofile = frame[y_d, xmin:xmax]
    yprofile = frame[ymin:ymax, x_d]
    
        
    fitParametersX, varianceMatrixX = scipy.optimize.curve_fit(gauss, 
                                                               range(len(xprofile)),
                                                               xprofile, p0 = [1000,10, 10],
                                                               bounds = ([0,0,0],[5000,20,np.inf]))
    fitParametersY, varianceMatrixY = scipy.optimize.curve_fit(gauss,
                                                               range(len(yprofile)),
                                                               yprofile, p0 = [1000,10, 10],
                                                               bounds = ([0,0,0],[5000,20,np.inf]))

    return np.mean([fitParametersX[2], fitParametersY[2]])
    

def getMaxPixelInDisk(frame, x, y, width):
    """
    This function outputs the largest pixel value in a box with size width at
    position x,y.
    """
    radius = 0.5*width
    x_d = int(np.round(x))
    y_d = int(np.round(y))
    maxPixelValue = 0
    #0.25 to put the entire pixel into the circle, instead of only the middle
    radius2 = (radius*radius-0.25)
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
    """
    This funciton outputs the sum of pixel values around x0, y0.
    """
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

def selectSameValues(image, value, pixelsInArea, nextPixels ,i0, j0): 
    """
    This function outputs the i0 and j0 value to pixelsInArea if the value in 
    the array image equals value. This function outputs each neighbouring x,y 
    positions around i0 and j0 and puts it in the array nextPixels.
    This function can be called again for these pixels, therefore acting as a
    todo list. This is needed because of the low max recursion depth in Python.
    """
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
                #selectSameValues(image, value, pixelsInArea, i1, j1)
                nextPixels.append([i1, j1])
                
def gauss(x, a, mu, sigma):
    """
    Value of Gaussian curve at x of amplitude a, mean mu and sigma sigma.
    """
    return a*np.exp(-(x-mu)**2/(2.0*sigma**2))





"""
All parameters and functions are defined. Now this script subtracts a image
convolved with a constant disk from a Gaussian blurred image. The pixels that 
have a pixel value above theshold are considdered signal. To remove small 
speckels an opening operation is performed that leaves only the larger blobs.
The binary image is dilated with a disk of similar size to capture all of the
signal.
"""


#Gaussian blur image
if(gaussianBlurRadius > 0):
    blurredImage = cv2.GaussianBlur(np.array(image), 
                                    (gaussianBlurRadius, gaussianBlurRadius), 0)
else:
    blurredImage = image.copy()

#The image is convolveld with a kernel of kernelDiskWidth +1 in size.

kernel = makeDiskShapedKernel(kernelDiskWidth+1)

#The dilation kernel has about the same size:
dilationKernel = makeDiskShapedKernel(kernelDiskWidth)

#The background is what is subtractd from the image.
backgroundImage = cv2.filter2D(np.array(image), -1, kernel/np.sum(kernel))



"""
The mask is the binary image after subtracting, thresholding, dilating and 
opening. mask is an array that has the same shape as the image. Each value is 0
if the pixel is SELECTED and 1 if the pixel is NOT selected (legacy reasons).
The output of this script are those regions of mask where the pixel value is 0
from the raw image. All other values are set to 0 (this supposedly makes further
analysis more easy.).

"""

masks = []




binary = []

mask = np.subtract(image, backgroundImage, dtype=np.int32)
mask = np.clip(mask, binaryThreshold, binaryThreshold+1)-binaryThreshold

binary = mask.copy()

openingKernel = makeDiskShapedKernel(6)
mask = cv2.morphologyEx(1.0*mask, cv2.MORPH_OPEN, np.array([1,1]*2))

mask = cv2.dilate(1.0*mask, np.uint8(dilationKernel), iterations = 1)

#makes all selected pixels 0 and non selected pixels 1. (legacy reasons.)
mask = 1 - mask

#this is the list of all not selected pixels. This is used to give some
#properties of the image. Together with some details about the signal, this can
#be extended to give automatically results about the image properties: 
#backlevel, SNR, percentage of background.
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

if(mpl.get_backend() != "Agg"):

    plt.figure()
    plt.imshow(np.array(image))
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


"""
The selection of all ROIs is done, now crop:
"""



imagesOfTraces = []
maskOfTraces = []

copyOfMask = mask.copy()
pixelsInArea = []

maxI = len(mask)
maxJ = len(mask[0])


"""

This loop loops over each pixel in the copy of mask. If the pixel equals value,
which should be 0.0, the pixels are added to the list. Then the neighbours are
checked if they have the same values. The cropped image is the rectangle that
contains all pixels. If the recursive blob finding is done, the pixels are set
to 1 and the loop continues until it finds a 0.0 again.

In the after cropping the cropped image is checked for the criterions set in
this script.

"""
for i in range(maxI):
    for j in range(maxJ):
        if(copyOfMask[i][j] == selectPixelsWithValue):
            pixelsInArea.append([i, j])

            #nextPixels works as a todo list to prevent recursion limit.
            nextPixels = []
            nextPixelsTemp = []
            selectSameValues(copyOfMask,
                             selectPixelsWithValue,
                             pixelsInArea,nextPixels,
                             i, j)
            
            while(nextPixels != []):
                newi, newj = nextPixels[-1]
                del nextPixels[-1]
                selectSameValues(copyOfMask,
                                 selectPixelsWithValue,
                                 pixelsInArea,nextPixels,
                                 newi, newj)

           
            
            if(len(pixelsInArea) < minArea):
                print("pixelsInArea not large enough")
                pixelsInArea = []
                continue
            p0 = [np.amin(np.array(pixelsInArea)[:,0]), np.amin(np.array(pixelsInArea)[:,1])]
            p1 = [np.amax(np.array(pixelsInArea)[:,0]), np.amax(np.array(pixelsInArea)[:,1])]
            
            #filter image:
            
            if((p0[0] < borderMargin)
               or (p0[1] < borderMargin)
               or (p1[0] > (len(copyOfMask) - borderMargin))
               or (p1[1] > (len(mask[0])-borderMargin))):
                print("too close to boundary")
                pixelsInArea = []

                continue
                       
            if( 
               (np.abs(p1[1] - p0[1]) < minWidth)
               or (np.abs(p1[1] - p0[1]) > maxWidth)
               or (np.abs(p1[0] - p0[0]) < minHeight)
               or (np.abs(p1[0] - p0[0]) > maxHeight)
               ):
                print("wrong shape:")
                print("%s, %s" % (p1[1] - p0[1], p1[0] - p0[0]))
                pixelsInArea = []
                continue
            
            print("Feature found that passed conditions.")
            
            trace = np.zeros((p1[0]-p0[0]+1,p1[1]-p0[1]+1))
            for pixel in pixelsInArea:
                trace[pixel[0]-p0[0]][pixel[1]-p0[1]] = np.array(image)[pixel[0]][pixel[1]]

            #trace = np.array(image)[p0[0]:p1[0], p0[1]:p1[1]].astype(np.int32())
            #trace = trace - 4095*np.array(mask)[p0[0]:p1[0], p0[1]:p1[1]].astype(np.uint16())
            #trace = np.clip(trace, 0, 65535).astype(np.uint16)
            imagesOfTraces.append(trace)  
            
            if(mpl.get_backend() != "Agg"):
                plt.figure()
                plt.imshow(np.array(mask)[p0[0]:p1[0],p0[1]:p1[1]])
                plt.show() 
            
                plt.figure()
                plt.imshow(trace)   
                plt.show()


            pixelsInArea = []

#plt.figure()
#plt.imshow(copyOfMask)
#plt.show()

del(copyOfMask)
print(len(imagesOfTraces)) 
               

"""

The cropping is done, now the length of the traces will be calculated by using
the higher intensity at the tips of the trace. As of writing (06-11-18) this
does not fully work on the test data.

"""
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
    
    """
    
    The traces are saved. 8 bit for viewing, 16 for analysing. 
    
    """
    
    exportImage = Image.fromarray(traceImage.astype(np.int32))
    exportImage.save(folderPath + "detectedTraces\\trace%d_16bit.png" % (traceNumber))
    exportImage = Image.fromarray(traceImage.astype(np.uint8))
    exportImage.save(folderPath + "detectedTraces\\trace%d_8bit.png" % (traceNumber))
    
    #The experimental track length feature output. Track length does not work yet:
    
    plt.figure()
    plt.imshow(traceImage)
    plt.plot([points[0][1], points[1][1]],[points[0][0], points[1][0]],"-wx")
    plt.savefig(folderPath + "detectedTraces\\processedTrace%d.png" % (traceNumber))
    plt.show()


np.savetxt(folderPath + "detectedTraces\\lengths.csv", distances, delimiter=",")
np.savetxt(folderPath + "detectedTraces\\angles.csv", angles, delimiter=",")
np.savetxt(folderPath + "detectedTraces\\points.csv", detectedPoints, delimiter=",")
           

