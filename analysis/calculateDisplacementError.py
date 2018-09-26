
# coding: utf-8

# In[363]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import pandas
import importlib.util


stepsPerFrame = 20

maxDistanceBetweenFrames = 3 #px

"""
TODO remove all steps with camerara exposed == False
"""

spec = importlib.util.spec_from_file_location("trackUtils.trackUtils", "D:\\Onderzoek\\git\\tracking\\trackUtils.py")
trackUtils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trackUtils)

pathToSimulatedLiks = "D:\\Onderzoek\\data\\simulation\\18-09-14\\tracks.csv"
pathToTrackedLinks = "D:\\Onderzoek\\data\\simulation\\18-09-14\\tracking\\18-09-26\\run5\\tracks0.csv"

simulatedLinks = pandas.read_csv(pathToSimulatedLiks)
trackedLinks = pandas.read_csv(pathToTrackedLinks) 
simulatedPositions = simulatedLinks.groupby(['frame',"particle"])["x", "y"].mean()
simulatedPositions.rename(columns={'x': 'x_simulated', 'y': 'y_simulated'}, inplace=True)

simulatedFrames = np.sort(np.unique(simulatedLinks['frame'].values))
trackedFrames = np.sort(np.unique(trackedLinks['frame'].values))
simPos = []
for i in simulatedFrames:
    simPos.append(simulatedPositions.loc[i].values)

    
    """ 
    swapped x and y, because input data was swapped. Change either trackParticles or simulation to change x and y.
    """ 
    
trackedPositions = trackedLinks.groupby(['frame',"particle"])["y", "x"].mean()
trackedPositions.rename(columns={'x': 'y_tracked', 'y': 'x_tracked'}, inplace=True)
trackedFrames = np.sort(np.unique(trackedLinks['frame'].values))
trackPos = []
for i in trackedFrames:
    trackPos.append(trackedPositions.loc[i].values)




# In[376]:


#Now get smallest distances:

for frame in trackedFrames:
    if frame in simulatedFrames:
        distances2 = np.zeros((len(trackPos[frame]), len(simPos[frame])))
        for i in range(len(trackPos[frame])):
            for j in range(len(simPos[frame])):
                distances2[i][j] = sum((trackPos[frame][i]-simPos[frame][j])**2)
        distances = []
        mapping = []
        
        for i, row in enumerate(distances2):
            minimum = np.amin(row)
            if(minimum <= maxDistanceBetweenFrames**2):
                minimum = np.sqrt(minimum)
                distances.append(minimum)
                mapping.append([i, np.argmin(row)])
            
errorPerFrame = np.mean(distances)
print("epsilon = %f px" % errorPerFrame )


# In[106]:





# In[377]:


a = simulatedPositions.groupby("particle").mean()


# In[378]:


b = trackedPositions.groupby("particle").mean()


# In[152]:





# In[379]:


distances2 = np.zeros((len(a), len(b)))
for i in range(len(distances2)):
    for j in range(len(distances2[0])):
        a.loc[i].tolist()
        distances2[i][j] = sum((np.array(a.loc[i]) - np.array(np.array(b.loc[j])))**2)
mapping = []
distances = []
for i, row in enumerate(distances2):
    minimum = np.amin(row)
    if(minimum <= maxDistanceBetweenFrames**2):
        minimum = np.sqrt(minimum)
        distances.append(minimum)
        mapping.append([i, np.argmin(row)])
        
mapping


# a

# In[380]:


mappingDict = {}
temp = {}
for e in mapping:
    temp.update({e[0] : -e[1]-1})
    mappingDict.update({-e[1]-1 : e[1]})
    
replaceDict = {'particle' : mappingDict}
print(mappingDict)

trackedPositions.rename(index=temp, inplace=True, level=1)
trackedPositions.rename(index=mappingDict, inplace=True, level=1)
trackedPositions.sort_index(inplace=True)


# In[381]:


print(simulatedPositions.iloc[0])
print(trackedPositions.iloc[0])


# In[398]:


allTrackedObjects = pandas.merge(simulatedPositions,trackedPositions, left_index=True, right_index=True )

pixelErrors = []
simulatedMeanSquareDisplacement = []
trackedMeanSquaredisplacement = []

for e in allTrackedObjects.iterrows():
    pixelErrors.append((e[1]['x_simulated']-e[1]['x_tracked'])**2+(e[1]['y_simulated']-e[1]['y_tracked'])**2)


meanPixelError = np.mean(pixelErrors)
meanPixelError





# In[397]:





# In[400]:


np.amin(pixelErrors)


# In[448]:


simulatedData = allTrackedObjects.groupby('particle')['x_simulated', 'y_simulated']
trackedData = allTrackedObjects.groupby('particle')['x_tracked', 'y_tracked']

simulatedMeanSquareDisplacement = []
trackedMeanSquaredisplacement = []

for i, particle in simulatedData:
    deltaFrames = np.diff(np.array(particle.index.tolist())[:,0])
    deltax2 = np.diff(particle['x_simulated'].tolist())**2
    deltay2 = np.diff(particle['y_simulated'].tolist())**2
    r2 = (deltax2 + deltay2)/deltaFrames
    simulatedMeanSquareDisplacement.append(np.mean(r2))


for i, particle in trackedData:
    deltaFrames = np.diff(np.array(particle.index.tolist())[:,0])
    deltax2 = np.diff(particle['x_tracked'].tolist())**2
    deltay2 = np.diff(particle['y_tracked'].tolist())**2
    r2 = (deltax2 + deltay2)/deltaFrames
    trackedMeanSquaredisplacement.append(np.mean(r2))
    
print(simulatedMeanSquareDisplacement)
print(trackedMeanSquaredisplacement)

np.mean(simulatedMeanSquareDisplacement) - np.mean(trackedMeanSquaredisplacement)
    

