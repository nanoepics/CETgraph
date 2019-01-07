# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:08:32 2018

@author: Peter
"""



import sys, os
import numpy as np
import math
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from PIL import Image
from PIL import ImageTk


folder = "D:\\Onderzoek\\data\\18-12-05\\afstand1\\200\\"
file = "exposedImageWihtoutBG.png"
runs = range(201,301,1)


maxWidth = 1000
maxHeight = 800
outputFolder = "D:\\Onderzoek\\data\\18-12-05\\200\\"

leftSelected = False
rightSelected = False

leftCorner = [0,0]
rightCorner = [0,0]
traces = [[]]


tkinterObject = tk.Tk()
tkinterObject.title("Koele klikker")

frames = []
tkFrames = []
pilFrames = []
for i in runs:
    path = folder + ("run%d" % (i)) + "\\" + file
    image = Image.open(path)
    image_8bit = np.array(image)
    image_8bit = (1.0/np.amax(image))*(1.0*image_8bit)
    image_8bit = Image.fromarray(np.uint8(image_8bit*255))
    imageSize = np.array(image).shape
    widthRatio = imageSize[1]/maxWidth
    heightRatio = imageSize[0]/maxHeight
    reshapeRatio = np.amax([widthRatio, heightRatio])
    newShape = (int(imageSize[1]/reshapeRatio), int(imageSize[0]/reshapeRatio))
    image_8bit = image_8bit.resize(newShape, Image.ANTIALIAS)
    
    
    frames.append(image_8bit)
    pilFrames.append(image)



imageNumber = 0


currentImage = frames[imageNumber]
tkImage = ImageTk.PhotoImage(currentImage, master=tkinterObject)

imageLabel = tk.Label(tkinterObject, compound = tk.LEFT,font = "25",
             text="%d"%(imageNumber), image=tkImage,fg = "white",
		 bg = "black")
imageLabel.pack()

def newImage():
    global imageNumber, leftSelected, rightSelected, imageLabel,tkImage, currentImage

    leftSelected = False
    rightSelected = False
    if(imageNumber +1 >= len(frames)):
        return
    traces.append([])
    imageNumber += 1
    currentImage = frames[imageNumber]
    tkImage = ImageTk.PhotoImage(currentImage, master=tkinterObject)
    imageLabel.config(text="%d"%(imageNumber), image=tkImage)

def undo():
    global leftSelected, rightSelected,currentImage, tkImage, imageLabel

    leftSelected = False
    rightSelected = False
    if(len(traces) <= imageNumber):
        return
    if(len(traces[-1]) <= 0):
        return    
    del traces[-1][-1]
    currentImage = np.array(frames[imageNumber])

    for trace in traces[-1]:
        leftCorner = trace[0]
        rightCorner = trace[1]
        currentImage[leftCorner[1]:rightCorner[1],leftCorner[0]:rightCorner[0]] = 255
        
    currentImage = Image.fromarray(currentImage)
    tkImage = ImageTk.PhotoImage(currentImage, master=tkinterObject)
    imageLabel.config(text="%d"%(imageNumber), image=tkImage)

def saveTraces():

    saveTraces = [np.multiply(reshapeRatio, row) for row in traces]

    np.savetxt(outputFolder + "traces.txt", saveTraces,fmt="%s", delimiter = ",")
    for n, tracesInFrame in enumerate(saveTraces):
        for i, trace in enumerate(tracesInFrame):
            
            croppedImage = pilFrames[n]
            trace[0][0] = int(np.clip(int(trace[0][0]), 0, croppedImage.size[0]-1))
            trace[0][1] = int(np.clip(int(trace[0][1]), 0, croppedImage.size[1]-1))
            trace[1][0] = int(np.clip(int(trace[1][0]), 0, croppedImage.size[0]-1))
            trace[1][1] = int(np.clip(int(trace[1][1]), 0, croppedImage.size[1]-1))
            croppedImage = croppedImage.crop((trace[0][0],trace[0][1],trace[1][0],trace[1][1]))
            croppedImage.save(outputFolder + 'im_%d_%d.png' % (n, i))
            print(outputFolder + 'im_%d_%d.png'% (n, i))

window = tk.Frame(tkinterObject)
window.pack()
nextButton = tk.Button(window,
                   text="Next",
                   command=newImage) 
nextButton.pack(side=tk.LEFT)

undoButton = tk.Button(window,
                   text="Undo",
                   command=undo)
undoButton.pack(side=tk.LEFT)

saveButton = tk.Button(window,
                   text="Save",
                   command=saveTraces) 
saveButton.pack(side=tk.LEFT)



def markSelection(corner1, corner2):
    global currentImage, tkImage, imageLabel
    temp = [np.amin([corner1[0], corner2[0]]),np.amin([corner1[1], corner2[1]]) ]
    rightCorner = [np.amax([corner1[0], corner2[0]]),np.amax([corner1[1], corner2[1]]) ]
    leftCorner = temp
    currentImage = np.array(currentImage)
    currentImage[leftCorner[1]:rightCorner[1],leftCorner[0]:rightCorner[0]] = 255
    currentImage = Image.fromarray(currentImage)
    tkImage = ImageTk.PhotoImage(currentImage, master=tkinterObject)
    imageLabel.config(text="%d"%(imageNumber), image=tkImage)
    
def saveTrace():
    global imageNumber, leftSelected, rightSelected, leftCorner, rightCorner
    if(not (leftSelected and rightSelected)):
        return

    leftSelected = False
    rightSelected = False
    
    
    temp = [np.amin([leftCorner[0], rightCorner[0]]),np.amin([leftCorner[1], rightCorner[1]]) ]
    rightCorner = [np.amax([leftCorner[0], rightCorner[0]]),np.amax([leftCorner[1], rightCorner[1]]) ]
    leftCorner = temp

    traces[-1].append([leftCorner, rightCorner])
    markSelection(leftCorner, rightCorner)



def selectTopLeftCorner(event):
    global leftSelected, leftCorner
    leftCorner = [event.x, event.y]
    leftSelected = True
    saveTrace()


def selectBottomRight(event):
    global rightSelected, rightCorner
    rightCorner = [event.x, event.y]
    rightSelected = True
    saveTrace()

def printXY(event):
    print("%d, %d" % (event.x, event.y)) 
    
    

    
    

imageLabel.bind('<Button-1>', selectTopLeftCorner)
imageLabel.bind('<Button-3>', selectBottomRight)


window.mainloop()












