# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:59:53 2018

@author: Peter
"""

"""
Thiss class provides different less scientific utility functions for camera.py,
trac.py and setMassBoundariesLog.py. It provides I/O functions and plotting
functions.

"""

import datetime  # date stamp
# from PIL import Image              # io images
import os  # io

import cv2  # export avi
import h5py  # import and export hdf5
import matplotlib.pyplot as plt  # plotting
import numpy as np  # numpy arrays
import trackpy as tp  # trackpy
from PIL import Image


class trackUtils:

    @staticmethod
    def createDirectoryTree(path, runs=0, foldername="", date=""):

        """
        This function creates directories for the date and for each run, so the data of each run is saved.
 
        input: root folder, returns number of runs
        """

        if (foldername == ""):
            foldername = "/tracking"
        if (date == ""):
            date = "/" + datetime.datetime.now().strftime("%y-%m-%d")
        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path + foldername):
            os.mkdir(path + foldername)

        if not os.path.isdir(path + foldername + date):
            os.mkdir(path + foldername + date)

        files = os.listdir(path + foldername + date)

        # get number of runs from directory name:
        if (runs == 0):
            for file in files:
                if (file[3:].isdigit() and file[0:3] == 'run' and int(file[3:]) > runs):
                    runs = int(file[3:])
            runs = runs + 1

        # make the folder for the output of this particular run:

        newPath = path + foldername + date + '/run' + str(runs)
        os.mkdir(newPath)
        print("Created Folder:\n" + newPath + "\n")
        return (runs, newPath)

    @staticmethod
    def loadData(path, subframes=[None], dataKey=None):

        print("Loadign h5 data from: " + path)
        file = h5py.File(path, 'r')  # open existing file in read-only mode
        if (dataKey == None):
            key = next(iter(file.keys()))  # getting the first key by default
        else:
            key = dataKey
        print(key + str(" dataset loaded."))
        data = file[key]  # corresponding dataset

        if (len(data.shape) == 3):
            print("Stack loaded. Number of frames:" + str(data.shape[0]) + " Image: " + str(data.shape[1]) + "x" + str(
                data.shape[2]))
        elif (len(data.shape) == 2):
            print("Image loaded. Image: " + str(data.shape[0]) + "x" + str(data.shape[1]))
        else:
            print("Wrong number of dimensions. (Colour data?)")
            print(data.shape)

        # each tracking object has its own set of variables, including frames:    
        frames = np.array(data)
        print(subframes)
        if (subframes != [None]):
            frames = frames[subframes[0]:subframes[1]]
        print("Data loaded")
        print("Pixel min/max: " + str(np.amin(data)) + "/" + str(np.amax(data)))

        file.close()

        return frames

    @staticmethod
    def getResults(trackingObject):

        outputText = "\n\n" + "number of detected particles: " + str(
            trackingObject.links['particle'].nunique()) + "\n" + \
                     "Particle diameter (nm): " + str(trackingObject.retrievedParticleDiameter) + "\n" + \
                     "Fit to all particles: n = " + str(trackingObject.trackpyFit[0][0]) + ", A: " + str(
            trackingObject.trackpyFit[0][1]) + "\n" + \
                     "Mean mobility y-direction: " + str(trackingObject.particleSpeed.mean()) + ' px/frame\n' + \
                     'Mean mobility y-direction: ' + str(trackingObject.mobility) + " um/s\n" + \
                     'Data from FFT:\nMean mobility y-direction: ' + str(
            4 * trackingObject.electricFrequency * trackingObject.amplitudeFFT) + ' px/s\n' + \
                     'Mean mobility y-direction: ' + str(trackingObject.mobilityFFT) + ' um/s\n' + \
                     'Frequency: ' + str(trackingObject.electricFrequency) + ' Hz \n' + \
                     'minimum frequency: ' + str(
            trackingObject.electricFrequency - trackingObject.window / trackingObject.cameraFPS) + ' Hz \n' + \
                     'maximum frequency: ' + str(
            trackingObject.electricFrequency + trackingObject.window / trackingObject.cameraFPS) + ' Hz \n' + \
                     'Amplitude ifft: ' + str(trackingObject.amplitudeFFT) + '\n'

        return outputText

    @staticmethod
    def getSettings(trackingObject):

        metadataText = "Run " + str(trackingObject.runs) + ' ' + str(datetime.datetime.now()) + "\n" + \
                       "Using data from file: " + trackingObject.framePath + "\n\nSettings:\n" + \
                       "Frames: " + str(trackingObject.subframes[0]) + " to " + str(
            trackingObject.subframes[1]) + "\n" + \
                       "particle diameter (px): " + str(trackingObject.particleDiameter) + "\n" + \
                       "minimum mass: " + str(trackingObject.minMass) + "\n" + \
                       "maximum mass: " + str(trackingObject.maxMass) + "\n" + \
                       "maximum excentricity: " + str(trackingObject.maximumExcentricity) + "\n" + \
                       "maximum distance between frames (px): " + str(trackingObject.maxTravelDistance) + "\n" + \
                       "minimum distance between frames (px): " + str(trackingObject.minimumMSD) + "\n" + \
                       "memory (frames): " + str(trackingObject.maxFrameMemory) + "\n" + \
                       "minimum number of frames of particle trajectory (frames): " + str(
            trackingObject.minimumParticleLifetime) + "\n" + \
                       "max lag time ( rames): " + str(trackingObject.maxLagTime) + \
                       "number of frames used (frames): " + str(trackingObject.useFrames) + "\n" + \
                       "number of frames of full data (frames): " + str(trackingObject.frameDimensions[0]) + "\n" + \
                       "Image size: " + str(trackingObject.frameDimensions[1]) + 'x' + str(
            trackingObject.frameDimensions[2]) + "\n" + \
                       "FPS: " + str(trackingObject.cameraFPS) + "\n" + \
                       "Exposure time (us): " + str(trackingObject.exposureTime) + "\n" + \
                       "Micron/pixel : " + str(trackingObject.micronPerPixel) + "\n"

        return metadataText

    @staticmethod
    def saveHDF5Data(data, key, dest):

        with h5py.File(dest, 'w') as hf:
            hf.create_dataset(key, data=data)
            hf.close()
        return

    @staticmethod
    def saveImage(data, dest, bit=8):

        if (bit == 8):
            data = (1.0 * data / np.amax(data)) * 255.0
            data = data.astype(np.uint8)
        figure = Image.fromarray(data)
        figure.save(dest)
        return

    @staticmethod
    def saveAVIData(data, dest, FPS, format='XVID', colour=False, logarithmic=False):
        if (logarithmic):
            trackUtils.saveAVIDataLogarithmic(data, dest, FPS, format, colour)
        else:
            trackUtils.saveAVIDataLinear(data, dest, FPS, format, colour)
        return

    @staticmethod
    def saveAVIDataLinear(data, dest, FPS, format='XVID', colour=False):
        # this function saves data as xvid avi format
        if (dest[-4:] is not ".avi"):
            dest = dest + ".avi"
        print("Saving to AVI: " + dest)
        maxPixelValue = np.amax(data)
        data = (1.0 * data / maxPixelValue) * 255.0
        data = np.array(data)
        data = data.astype(np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*format)
        videoFile = cv2.VideoWriter(dest, fourcc, FPS, (len(data[0][0]), len(data[0])), colour)

        for i in range(np.shape(data)[0]):
            videoFile.write(np.uint8(data[i, :, :]))

        videoFile.release()
        print("AVI saved")
        return

    @staticmethod
    def saveAVIDataLogarithmic(data, dest, FPS, format='XVID', colour=False):
        # this function saves data as xvid avi format

        if ("." not in dest or dest[-3] is not "."):
            dest = dest + ".avi"
        print("Saving to AVI: " + dest)
        data = data + 1
        data = np.log(data)
        maxPixelValue = np.amax(data)
        data = (1.0 * data / maxPixelValue) * 255.0
        data = np.array(data)
        data = data.astype(np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*format)
        videoFile = cv2.VideoWriter(dest, fourcc, FPS, (len(data[0][0]), len(data[0])), colour)

        for i in range(np.shape(data)[0]):
            videoFile.write(np.uint8(data[i, :, :]))

        videoFile.release()
        return

    @staticmethod
    def loadFromInstructionFile(text, path):
        try:
            f = open(path, "r")
            input = f.read().split("\n")
            f.close()
            for element in input:
                element = element.split(',')
                if (len(element) < 2):
                    continue
                element[1] = element[1].replace("'", "")
                element[1] = element[1].replace(" ", "")
                if (element[0] == text):
                    return element[1]

            return -1
        except:
            print("Cannot find instruction file.")
            return -9999

    @staticmethod
    def getFromMetaData(text, path):
        """
        This function searches a metadata file generated by camera script
        in path to get stored data form that file. return -1 if failed, but file exists and -9999 if
        file does not exist:
        """
        try:
            f = open(path, "r")
            input = f.read().split("\n")
            f.close()
            for element in input:
                element = element.split(',')
                if (len(element) < 2):
                    continue
                element[0] = element[0][2:-1]
                element[1] = element[1].replace("'", "")
                element[1] = element[1][:-1]
                if (element[0] == text):
                    return element[1].strip()
            print("Entry not in metadata file.")
            return -1
        except:
            print("Cannot find metadata file.")
            return -9999

    @staticmethod
    def writeMetadata(text, dest):
        try:
            f = open(dest, "a")
            f.write(text)
            f.close()
        except:
            print("Cannot make metadata file.")
        return

    @staticmethod
    def writeOutputToFolder(folder, trackingObject, metadataFile="metadata.txt"):

        trackUtils.writeMetadata(trackUtils.getSettings(trackingObject), folder + "/metadata.txt")
        trackUtils.writeMetadata(trackUtils.getResults(trackingObject), folder + "/metadata.txt")

        np.savetxt(folder + "\\diffusionConstant" + str(trackingObject.runs) + ".csv",
                   trackingObject.diffusionConstants, delimiter=",")
        np.savetxt(folder + "\\particleDiameters" + str(trackingObject.runs) + ".csv", trackingObject.particleDiameters,
                   delimiter=",")
        np.savetxt(folder + "\\massDistribution.csv", trackingObject.links['mass'], delimiter=",")
        np.savetxt(folder + "\\pathLengths" + str(trackingObject.runs) + ".csv", trackingObject.visibleInFrames,
                   delimiter=",")
        np.savetxt(folder + "\\eccentricity" + str(trackingObject.runs) + ".csv", trackingObject.eccentricity,
                   delimiter=",")
        np.savetxt(folder + "\\eccentricityWeights" + str(trackingObject.runs) + ".csv",
                   trackingObject.eccentricityWeights, delimiter=",")
        np.savetxt(folder + "\\mobility" + str(trackingObject.runs) + ".csv", trackingObject.averageMobilities,
                   delimiter=",")
        np.savetxt(folder + "\\mobilityWeights" + str(trackingObject.runs) + ".csv",
                   trackingObject.averageMobilityWeights, delimiter=",")
        return

    @staticmethod
    def generateMiscPlots(trackingObject, binsize=20, plotMinDiameter=0, plotMaxDiameter=200, maxLagTime=None,
                          silent=False):

        """
        This function generated various plots and writes tracking parameters to metadata.txt.
        """

        if (maxLagTime == None):
            maxLagTime = trackingObject.maxLagTime

        plt.figure()
        figure = plt.gcf()
        figure = trackingObject.drift.plot()
        plt.savefig(trackingObject.currentPath + '/drift' + str(trackingObject.runs) + '.pdf')
        # plt.show(figure)

        powerLaw = tp.emsd(trackingObject.links, trackingObject.micronPerPixel, trackingObject.cameraFPS,
                           max_lagtime=maxLagTime)

        plt.figure()
        figure = plt.gcf()
        plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
        plt.xlabel('lag time $t$');
        trackpyFit = tp.utils.fit_powerlaw(powerLaw)
        figure.savefig(trackingObject.currentPath + '/powerLawFit' + str(trackingObject.runs) + '.pdf')

        print(trackingObject.diffusionConstants)

        plt.figure()
        n, bins, patches = plt.hist(trackingObject.diffusionConstants[:, 1], binsize)
        figure = plt.gcf()
        plt.xlabel('D')
        plt.ylabel('Count')
        plt.title('Histogram of Diffusion Constants (individual partilces)')
        plt.grid(True)
        figure.savefig(trackingObject.currentPath + '/diffusionConstants' + str(trackingObject.runs) + '.pdf')
        # plt.show()

        plt.figure()
        n, bins, patches = plt.hist(trackingObject.diffusionConstants[:, 1], binsize,
                                    weights=trackingObject.visibleInFrames)
        figure = plt.gcf()
        plt.xlabel('D')
        plt.ylabel('Count')
        plt.title('Histogram of Diffusion Constants')
        plt.grid(True)
        figure.savefig(trackingObject.currentPath + '/diffusionConstantsWeighted' + str(trackingObject.runs) + '.pdf')
        # plt.show()

        plt.figure()
        n, bins, patches = plt.hist(trackingObject.particleDiameters[:, 1], binsize)
        figure = plt.gcf()
        plt.xlabel('a (nm)')
        plt.ylabel('Count')
        plt.title('Histogram of Diameters (individual particles)')
        plt.grid(True)
        figure.savefig(trackingObject.currentPath + '/diameters' + str(trackingObject.runs) + '.pdf')
        # plt.show()

        plt.figure()
        n, bins, patches = plt.hist(trackingObject.particleDiameters[:, 1], binsize,
                                    weights=trackingObject.visibleInFrames)
        figure = plt.gcf()
        plt.xlabel('a (nm)')
        plt.ylabel('Count')
        plt.title('Histogram of Diameters')
        plt.grid(True)
        figure.savefig(trackingObject.currentPath + '/diametersWeighted' + str(trackingObject.runs) + '.pdf')
        # plt.show()

        plt.figure()
        n, bins, patches = plt.hist(trackingObject.eccentricity, weights=trackingObject.eccentricityWeights)
        figure = plt.gcf()
        plt.xlabel('Eccentricity')
        plt.ylabel('Fraction')
        plt.title('Histogram of eccentricities')
        plt.grid(True)
        figure.savefig(trackingObject.currentPath + '/eccentricitiesWeighted' + str(trackingObject.runs) + '.pdf')

        plt.figure()
        n, bins, patches = plt.hist(trackingObject.visibleInFrames, binsize)
        figure = plt.gcf()
        plt.xlabel('Path length in frames')
        plt.ylabel('Count')
        plt.title('Histogram of path length')
        plt.grid(True)
        figure.savefig(trackingObject.currentPath + '/pathLengthHistogram' + str(trackingObject.runs) + '.pdf')
        # plt.show()

        plt.figure()
        n, bins, patches = plt.hist(
            [e for e in trackingObject.particleDiameters[:, 1] if ((e >= plotMinDiameter) and (e <= plotMaxDiameter))],
            binsize)
        figure = plt.gcf()
        plt.xlabel('a (nm)')
        plt.ylabel('Count')
        plt.title('Histogram of Diameters')
        plt.grid(True)
        figure.savefig(trackingObject.currentPath + '/diametersZoom' + str(trackingObject.runs) + '.pdf')
        # plt.show()

        plt.figure()
        fig, ax = plt.subplots()
        ax.hist(trackingObject.links['mass'], bins=20)
        ax.set(xlabel='mass', ylabel='count')
        plt.grid(True)
        plt.savefig(trackingObject.currentPath + '/massDistributionFiltered_run' + str(trackingObject.runs) + '.pdf')

        # plt.show()

        masslist = []
        # print(self.links.groupby('particle').mean())
        massDiffusion = []
        massDiameter = []
        for particleID in list(trackingObject.msdData):
            mass = trackingObject.links.loc[trackingObject.links['particle'] == particleID]['mass'].mean()
            if (particleID in trackingObject.diffusionConstants[:, 0]):
                masslist.append([particleID, mass])
                indexOfID = list(trackingObject.diffusionConstants[:, 0]).index(particleID)
                diffusion = trackingObject.diffusionConstants[:, 1][indexOfID]
                diameter = trackingObject.particleDiameters[:, 1][indexOfID]
                massDiffusion.append([mass, diffusion])
                massDiameter.append([mass, diameter])
        massDiffusion = np.array(massDiffusion)
        massDiameter = np.array(massDiameter)

        plt.figure()
        plt.scatter(massDiffusion[:, 0], massDiffusion[:, 1])
        figure = plt.gcf()
        plt.xlabel("Mass")
        plt.ylabel("Diffusion Constant")
        plt.title("Diffusion versus Mass")
        plt.grid(True)
        figure.savefig(trackingObject.currentPath + '/MassDiffusionScatterplot' + str(trackingObject.runs) + '.pdf')
        # plt.show()

        plt.figure()
        plt.scatter(massDiameter[:, 0], massDiameter[:, 1])
        figure = plt.gcf()
        plt.xlabel("Mass")
        plt.ylabel("Particle Diameter (nm)")
        plt.title("Diameter versus Mass")
        plt.grid(True)
        figure.savefig(trackingObject.currentPath + '/MassDiameterScatterplot' + str(trackingObject.runs) + '.pdf')
        # plt.show()

        return
