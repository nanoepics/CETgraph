import matplotlib.pyplot as plt 
import sys, os,csv
import numpy as np
plt.switch_backend("Agg")

frameWindow = -1
date = "01-06-18"
maxDiameter = 200
maxMassZoom = 2000
run = 1

if(len(sys.argv)>1):
   path = sys.argv[1]
else:
   path = "D:\\Onderzoek\\data\\01-06-2018\\run8\\500\\tracking\\03-06-18\\run" + str(run)

outputPath = path



plt.rc('font',size=14)





particleDiameters = []
pathLengths = []
diffusionConstant = []
if(len(sys.argv)>2):
   runInt = int(sys.argv[2])
else:
   runInt = run
   if(len(sys.argv)>1):
      for i in range(len(path)-1,0,-1):
         if(not path[i].isdigit()):
            runString = path[i+1:]
            runInt = int(runString)
            break

diffustion = np.genfromtxt(path + "\\diffusionConstant" +str(runInt) + ".csv",delimiter=',')[:,1]
diffusionConstant.extend(diffustion)
print('diffusion data loaded')
diameter = np.genfromtxt(path + "\\particleDiameters" +str(runInt) + ".csv",delimiter=',')[:,1]
print('diameter data loaded')
particleDiameters.extend(diameter)
pathLength = np.genfromtxt(path + "\\pathLengths" +str(runInt) + ".csv",delimiter=',')
pathLengths.extend(pathLength)
print('path length data loaded')
massDistribution = np.genfromtxt(path + "\\massDistribution.csv",delimiter=',')
print('particle intensity data loaded')

print(len(massDistribution))

weights = pathLengths/sum(pathLengths)

diameterZoom = []
weightsZoom = []
weightsMassZoom = []
massZoom = []

for i in range(len(particleDiameters)):
   if(particleDiameters[i] > 0 and particleDiameters[i] <= maxDiameter ):
      diameterZoom.append(particleDiameters[i])
      weightsZoom.append(weights[i])

for i in range(len(massDistribution)):
   if(massDistribution[i] > 0 and massDistribution[i] <= maxMassZoom):
      massZoom.append(massDistribution[i])
      weightsMassZoom .append(weights[i])

weightsZoom = weightsZoom/sum(weightsZoom)
weightsMassZoom = weightsMassZoom/sum(weightsMassZoom)
diameterPlotMaximum = int(np.amax(particleDiameters))
if (diameterPlotMaximum >5000):
   diameterPlotMaximum = 5000
diameterPlotBinSize = int(diameterPlotMaximum/20)




print("plotting path lengths")
plt.figure()
n, bins, patches = plt.hist(pathLengths, bins=range(0,int(np.amax(pathLengths)),8), facecolor = '#0000ff')
figure = plt.gcf()
plt.xlabel('Path length (frames)')
plt.ylabel('Count')
plt.title('Histogram of Path Lengths')
plt.grid(True)
figure.savefig(outputPath + '//pathLength.pdf')

print("plotting mass")
plt.figure()
n, bins, patches = plt.hist(massDistribution, bins=range(0,int(np.amax(massDistribution)),100), facecolor = '#0000ff',weights=weights)
figure = plt.gcf()
plt.xlabel('Mass')
plt.ylabel('Count')
plt.title('Histogram of Intensity')
plt.grid(True)
figure.savefig(outputPath + '//massDistribution.pdf')

print("plotting mass (up to " + str(maxMassZoom) + ")" )
plt.figure()
n, bins, patches = plt.hist(massZoom, bins=range(0,maxMassZoom,20), facecolor = '#0000ff',weights = weightsMassZoom)
figure = plt.gcf()
plt.xlabel('Mass')
plt.ylabel('Count')
plt.title('Histogram of Intensity')
plt.grid(True)
figure.savefig(outputPath + '//massDistributionZoom.pdf')

print("plotting diameters")
plt.figure()
n, bins, patches = plt.hist(particleDiameters,  bins=range(0,diameterPlotMaximum ,diameterPlotBinSize), facecolor = '#0000ff')
figure = plt.gcf()
plt.xlabel('a (nm)')
plt.ylabel('Count')
plt.title('Histogram of Diameters')
plt.grid(True)
figure.savefig(outputPath + '//diameters.pdf')

print("plotting path diffusion")
plt.figure()
n, bins, patches = plt.hist(diffusionConstant,40, facecolor = '#0000ff',weights=weights)
figure = plt.gcf()
plt.xlabel('D (um^2/s)')
plt.ylabel('Fraction')
plt.title('Histogram of Diffusion Constant')
plt.grid(True)
figure.savefig(outputPath + '//diffusion.pdf')

print("plotting diameters (up to 200)")
plt.figure()
n, bins, patches = plt.hist(diameterZoom , bins = range(0,208,8), facecolor = '#0000ff' ,weights = weightsZoom )
figure = plt.gcf()
plt.xlabel('a (nm)')
plt.ylabel('Fraction')
plt.title('Histogram of Diameters')
plt.grid(True)
figure.savefig(outputPath + '//diametersZoom.pdf')



