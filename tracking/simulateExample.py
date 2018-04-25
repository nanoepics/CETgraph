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


noise = 30
signalPerSpecies = [22,23,24,25,26]
fov = [750,600]
difcon = 0
numberOfFrames = 500
signal = 100
noise = 10
psize = 45
useRandomIntensity = False
numberPerSpecies = [1,1,1,1,1]
pixelSizePerSpecies = [20,20,20,20,20]
staticNoise = 0.0
FPS = 40
micronPerPixel = 0.225664
particleDiameter = 100

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




def getMetadata():
    tableNumberPerSpecies = ""
    tableSignalPerSpecies = ""
    tablePixelSizePerSpecies = ""
    for i in range(len(numberPerSpecies)):
        tableNumberPerSpecies = tableNumberPerSpecies +  str(numberPerSpecies[i]) + " ,"
        tableSignalPerSpecies = tableSignalPerSpecies + str(signalPerSpecies[i]) + " ,"
        tablePixelSizePerSpecies =  tablePixelSizePerSpecies +str(pixelSizePerSpecies[i]) + " ,"
    
    tableNumberPerSpecies = tableNumberPerSpecies[:-2]
    tableSignalPerSpecies = tableSignalPerSpecies[:-2]
    tablePixelSizePerSpecies = tablePixelSizePerSpecies[:-2]
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
    "Static noise: " + str(staticNoise) + "\n" + \
    "Micron per pixel: " + str(micronPerPixel) + "\n" + \
    "FPS: " + str(FPS) + "\n" + \
    "particle diameter (nm): " + str(particleDiameter)
    return metadata



print("Start run: " + str(runs))


initialRuns = runs



folder = currentPath + "\ParticleDiameter" + str(particleDiameter)+ "_noise_" + str(noise) + "_FPS_" + str(FPS)
os.mkdir(folder)




setup = SingleFrame(fov=fov, signal=signal, noise = noise, psize = psize, useRandomIntensity = useRandomIntensity, numberPerSpecies = numberPerSpecies, signalPerSpecies = signalPerSpecies, pixelSizePerSpecies =  pixelSizePerSpecies, staticNoise = staticNoise,FPS = FPS, micronPerPixel = micronPerPixel,particleDiameter = particleDiameter)
stack = setup.genStack(numberOfFrames)
stack =  np.swapaxes(stack,0,2)
difcon = setup.difcon
writeMetadata(getMetadata(),folder=folder)

with h5py.File(folder + '/data.h5', 'w') as hf:
    hf.create_dataset("Simulated data",  data=stack)
    hf.close()

stack =  np.swapaxes(stack,0,2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoFile = cv2.VideoWriter(folder + '/simulated data.avi',fourcc,FPS,(len(stack[0]),len(stack)),False)
for i in range(np.shape(stack)[2]):
    videoFile.write(np.uint8(stack[:,:,i]))
videoFile.release()







