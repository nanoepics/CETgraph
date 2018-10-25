# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:36:49 2018

@author: Peter
"""
import sys, os
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import simulateBrownianMotion
from trackUtils import trackUtils


if(len(sys.argv) == 1):
    outputPath = "D:\\Onderzoek\\data\\simulation\\18-10-05\\"
else:
    outputPath = sys.argv[1]

if(len(sys.argv) > 1):
    numberOfFrames = int(sys.argv[2])
else:
    numberOfFrames = 1000



FPS = 55
stepsPerFrame = 100
stepsExposed = 100

particleSignal = 1700 / stepsExposed

exposureTime = stepsExposed/(FPS*stepsPerFrame)
deltat = 1/(FPS*stepsPerFrame)


print("FPS: %f, Frame Time: %f, exposure Time: %f, step delta t: %f, \
      steps per frame: %d, steps exposed: %d" %
      (FPS, 1/FPS, exposureTime, deltat, stepsPerFrame, stepsExposed))


# required: FOV, deltat, micronPerPixel 
#optional: temperature = 293, viscosity = 0.001, margin = 20

simulation = simulateBrownianMotion.BrownianSimulation((300, 300), deltat, 0.225664, 
                                                       noiseBackLevel = 225.0, 
                                                       gaussianNoiseLevel = 25.0)


simulation.MARGIN = 130
simulation.addParticle(100, particleSignal, 2.5, PSFSize = (15,15))
simulation.MARGIN = 10

simulation.particles[0].particleID

simulation.generateNewImage()

#simulateBatch(maxFrames, stepsPerFrame, stepsExposed):

simulation.simulateBatch(numberOfFrames, stepsPerFrame, stepsExposed)

#The frames simulated are in floats. Bin to ints to make data comparable to
#experiment. How this binning is being done depends on the particular settings
#used in experiment. The data needs to
#be binned to 8-bit for the AVI. 

maximumIntensity = np.amax(simulation.frames)



u12Frames = np.array([np.uint16(np.clip((frame),0, 4095)) for frame in simulation.frames])
u8Frames = np.array([np.uint8((frame/maximumIntensity)*255) for frame in simulation.frames])

trackUtils.saveHDF5Data(u12Frames, "Simulated data", outputPath + "data.h5")

trackUtils.saveAVIData(u8Frames, outputPath + "simulation", FPS)
trackUtils.saveAVIData(u8Frames, outputPath + "simulationLog", FPS, logarithmic=True)

"""


with open(outputPath + "tracks.csv", "w") as file:
    for frame in simulation.tracks:
        for row in frame:
            file.write('%d, %d, %f, %f' % row)
            file.write("\r\n")
"""

simulation.tracks
tracks = pd.DataFrame(data = simulation.tracks,
                      index = range(len(simulation.tracks)),
                      columns=["frame", "step", "particle", "x", "y", "exposure on"])
tracks.to_csv(outputPath + "tracks.csv")

with open(outputPath + "particleProperties.csv", "w") as file:
    for particle in simulation.particles:
        properties = particle.__dict__.items()
        for variable, value in properties:
            file.write("['%s', %s], " % (variable, value))
        file.write("\n")



with open(outputPath + "simulationProperties.csv", "w") as file:
    file.write("FPS: %f, Frame Time: %f, exposure Time: %f, step delta t: %f, \
               steps per frame: %d, steps exposed: %d" %
               (FPS, 1/FPS, exposureTime,
                deltat, stepsPerFrame, stepsExposed))




