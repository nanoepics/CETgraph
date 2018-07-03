"""
    CETgraph.examples.simulate2Dstack.py
    ==================================
    This example, generates a synthetic stack of 2d frames using the .simulate class.
    generally, a well prepared analysis code only handles the input and output data. All other generic calculations
    should be done via the methods in other CETgraph modules and libraries.
    .. lastedit:: 1/3/2018
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""

from tracking.simulate import SingleFrame

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import h5py
import os
import datetime
import trackpy as tp
import scipy.misc
from PIL import Image


noise = 2
signalPerSpecies = [10,10]
fov = [250,300]
difcon = 0
numberOfExposures = 5
exposureTime = -1
signal = 200
noise = 2
psize = 10
useRandomIntensity = False
numberPerSpecies = [64,0]
pixelSizePerSpecies = [3,3]
staticNoise = 0.0
FPS = 55
micronPerPixel = 0.75374991
particleDiameters = [100,100]
electricField  = [0,3]
electrophoreticMobility = 1
electricFrequency = 0.5
electrophoreticMobilityPerSpecies = [10,1]
numberOfFrames = 200
exposureFrequency = -1
signalType = 'sin'


if(exposureTime < 0):
    exposureTime = (1/(numberOfExposures*FPS))


if(exposureFrequency < 0):
    exposureFrequency = 1/exposureTime
print("Exposure time: " + str(exposureTime*numberOfExposures) + " s, time per frame: " + str(1/FPS) + " s." )



detectedParticles = []
today = datetime.datetime.now().strftime("%d-%m-%y")
runs = 0
currentPath = './simulation/runs/' + today + '/run' + str(runs)

def writeMetadata(text, folder = currentPath,file = "metadata.txt"):
    try:
        print("Writing to metadata.txt")
        f = open(folder + '/'+ file, "a")
        f.write(text)
        f.close()
    except:
        print("Cannot make metadata file.")
        

    

if not os.path.isdir("./simulation"):
    os.mkdir("./simulation")        
        
if not os.path.isdir("./simulation/runs"):
    os.mkdir("./simulation/runs")
    
if not os.path.isdir("./simulation/runs/" + today ):
    os.mkdir("./simulation/runs/" + today)


files = os.listdir("./simulation/runs/" + today)



#get number of runs:
for file in files:
    if(file[3:].isdigit() and file[0:3] == 'run' and int(file[3:]) > runs):
        runs = int(file[3:])
runs  = runs + 1

currentPath = './simulation/runs/' + today + '/run' + str(runs)

os.mkdir(currentPath)
folder = currentPath



def getMetadata():
    tableNumberPerSpecies = ""
    tableSignalPerSpecies = ""
    tablePixelSizePerSpecies = ""
    tableDiameterPerSpecies = ""
    tableElectrophoreticMobilityPerSpecies = ""
    for i in range(len(numberPerSpecies)):
        tableNumberPerSpecies = tableNumberPerSpecies +  str(numberPerSpecies[i]) + " ,"
        tableSignalPerSpecies = tableSignalPerSpecies + str(signalPerSpecies[i]) + " ,"
        tablePixelSizePerSpecies =  tablePixelSizePerSpecies +str(pixelSizePerSpecies[i]) + " ,"
        tableDiameterPerSpecies = tableDiameterPerSpecies + str(particleDiameters[i]) + " ,"
        tableElectrophoreticMobilityPerSpecies = tableElectrophoreticMobilityPerSpecies + str(electrophoreticMobilityPerSpecies[i]) + " ,"
    tableNumberPerSpecies = tableNumberPerSpecies[:-2]
    tableSignalPerSpecies = tableSignalPerSpecies[:-2]
    tablePixelSizePerSpecies = tablePixelSizePerSpecies[:-2]
    tableDiameterPerSpecies = tableDiameterPerSpecies[:-2]
    tableElectrophoreticMobilityPerSpecies  = tableElectrophoreticMobilityPerSpecies [:-2]
    metadata =  "Run " + str(runs) +  ' ' + str(datetime.datetime.now()) + "\n" + \
    "Number of frames: " + str(numberOfFrames) + "\n" + \
    "Field of view: "  + str(fov[0]) + " x "+ str(fov[1]) + "\n"  + \
    "Diffusion constant: " + str(difcon) + "\n" + \
    "Signal: " + str(signal) + "\n" + \
    "Noise: " + str(noise) + "\n" + \
    "Pixel size particle: " + str(psize) + "\n" + \
    "Random particle intensity used: " + str(useRandomIntensity) + "\n" + \
    "Number per species: " + tableNumberPerSpecies + "\n" + \
    "Signal per species: " + tableSignalPerSpecies + "\n" + \
    "Pixel Size per species: " + tablePixelSizePerSpecies + "\n" + \
    "Diameters per particle: " + tableDiameterPerSpecies + "\n" + \
    "Electrophoretic mobility per particle: " + tableElectrophoreticMobilityPerSpecies + "\n" + \
    "Electric Field: (" + str(electricField[0]) + ", " + str(electricField[1]) + ")" + \
    "Static noise: " + str(staticNoise) + "\n" + \
    "Electric frequency: " + str(electricFrequency) + "\n" + \
    "Micron per pixel: " + str(micronPerPixel) + "\n" + \
    "FPS: " + str(FPS) + "\n" + \
    "Signal type: " + signalType + "\n"  + \
    "Total exposure time: " + str(exposureTime*numberOfExposures) + "\n" + \
    "Exposure time per exposure: " + str(exposureTime) + "\n" + \
    "Number of exposures per frame: " + str(numberOfExposures) + "\n\n\n" + \
    "Relevant data in inputform for track.py:\n\n" + \
    "['ResultingFrameRate', " + str(FPS) + "]\n" + \
    "['PixelSize', " + str(micronPerPixel) + "]\n" 
    
    
    
    return metadata



print("Start run: " + str(runs))


initialRuns = runs






setup = SingleFrame(fov=fov, signalType = signalType,electrophoreticMobilityPerSpecies = electrophoreticMobilityPerSpecies,electricFrequency = electricFrequency, backgroundIntensity = 0,electricField  = electricField, electrophoreticMobility = electrophoreticMobility, signal=signal, noise = noise, psize = psize, useRandomIntensity = useRandomIntensity, numberPerSpecies = numberPerSpecies, signalPerSpecies = signalPerSpecies, pixelSizePerSpecies =  pixelSizePerSpecies, staticNoise = staticNoise,FPS = FPS, micronPerPixel = micronPerPixel,particleDiameters = particleDiameters)
#stack = setup.genStack(numberOfFrames)
stack = setup.genMultipleExposedImage(numberOfExposures =numberOfExposures,nframes = numberOfFrames, exposureFrequency = exposureFrequency)

np.savetxt(folder +"\\electricSignal.csv", setup.electricFieldData,delimiter=",")


if(len(np.shape(stack))>2):
   stack =  np.swapaxes(stack,0,2)
difcon = setup.difcon
writeMetadata(getMetadata(),folder=folder)

with h5py.File(folder + '/data.h5', 'w') as hf:
    hf.create_dataset("Simulated data",  data=stack)
    hf.close()
    

print(np.shape(stack))


if(len(np.shape(stack))>2):
   stack =  np.swapaxes(stack,0,2)
   fourcc = cv2.VideoWriter_fourcc(*'XVID')
   videoFile = cv2.VideoWriter(folder + '/simulated data.avi',fourcc,FPS,(len(stack[0]),len(stack)),False)
   for i in range(np.shape(stack)[2]):
      videoFile.write(np.uint8(stack[:,:,i]))
   videoFile.release()
   scipy.misc.imsave(folder + '/overexposedImage.png', np.uint8(stack[:,:,0]))

elif(len(np.shape(stack))==2):
   scipy.misc.imsave(folder + '/overexposedImage.png', np.uint8(stack))








