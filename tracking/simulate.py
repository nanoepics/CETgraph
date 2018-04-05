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


batchJobNoiseList = range(0,50,2)
batchJobSignalList = [(ii,jj) for (ii,jj) in zip(np.full(6,50),range(45,51,1))]

print(batchJobNoiseList)
print(batchJobSignalList)

batchJobNoiseList = [30]
batchJobSignalList = [[50,48]]


fov = [100,100]
numberOfFrames = 1
difcon = 5
signal = 100
noise = 10
psize = 45
useRandomIntensity = False
numberPerSpecies = [12,12]
signalPerSpecies = [5,15]
pixelSizePerSpecies = [20,20]
staticNoise = 0.0

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
    "Static noise: " + str(staticNoise) + "\n"
    return metadata



print("Start run: " + str(runs))


initialRuns = runs

for signalPerSpecies in batchJobSignalList:
    for noise in batchJobNoiseList:

        folder = currentPath + "\DeltaSignal_" + str(abs(signalPerSpecies[0] - signalPerSpecies[1]))+ "_noise_" + str(noise)
        os.mkdir(folder)

        print(str(int((runs-initialRuns)*100/(len(batchJobSignalList)*len(batchJobNoiseList)))) + "%")
        
        writeMetadata(getMetadata(),folder=folder)
        setup = SingleFrame(fov=fov, difcon=difcon, signal=signal, noise = noise, psize = psize, useRandomIntensity = useRandomIntensity, numberPerSpecies = numberPerSpecies, signalPerSpecies = signalPerSpecies, pixelSizePerSpecies =  pixelSizePerSpecies, staticNoise = staticNoise)
        stack = setup.genStack(numberOfFrames)

        
        with h5py.File(folder + '/simulated data.h5', 'w') as hf:
            hf.create_dataset("Simulated data",  data=stack)
            hf.close()
        
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videoFile = cv2.VideoWriter(folder + '/simulated data.avi',fourcc,40,(len(stack[0]),len(stack)),False)
        for i in range(np.shape(stack)[2]):
            videoFile.write(np.uint8(stack[:,:,i]))
        videoFile.release()
        
        particles = tp.locate(stack[:,:,0], 35, minmass = min(signalPerSpecies)*200)
        plt.figure()
        figure = plt.gcf()
        tp.annotate(particles, stack[:,:,0])
        figure.savefig(folder + '/detectedParticlesRun' + str(runs) + '.png')
        
        detectedParticles.append([signalPerSpecies[0],signalPerSpecies[1],noise,len(particles)])

        writeMetadata("Number of particles found: " + str(len(particles)),folder=folder)
        writeMetadata("\nFound particles:\n\n",folder=folder)
        writeMetadata(str(particles),folder=folder)
        writeMetadata(str(particles),folder=folder, file = 'particles.dat')     
        writeMetadata(str(particles['mass']),folder=folder, file = 'mass.dat')        
        
        fig, ax = plt.subplots()
        ax.hist(particles['mass'], bins=20)
        ax.set(xlabel='mass', ylabel='count')
        plt.savefig(folder + '/massDistributionRun' + str(runs) + '.pdf')







detectedParticles = np.array(detectedParticles)
plt.figure()


def f(x):
    if(x < 0):
        return -1
    elif(x > 0):
        return 1
    return 0

cmap = matplotlib.colors.ListedColormap(['#ff0000','#00ff00','#0000ff'])


f = np.vectorize(f)
kleur = f(detectedParticles[:,3]-np.sum(numberPerSpecies))

plt.scatter(detectedParticles[:,0]-detectedParticles[:,1], detectedParticles[:,2], c = kleur,cmap=cmap)
plt.xlabel('$\Delta$I')
plt.ylabel('Noise')
plt.savefig(currentPath + '/scatterplotDetectedParticles' + str(runs) + '.pdf')
plt.show()


