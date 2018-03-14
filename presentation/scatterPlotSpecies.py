# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 18:01:41 2018

@author: Peter
"""


import matplotlib.pyplot as plt





file = "D:/Onderzoek/CETgraph-dashka/CETgraph-dashka/histogram/14-03-18/output.dat"

signal = []
noise = []
dist = []


def getData(file):
    try:
        output = []
        #print("Reading " + file)
        f = open(file, "r")
        input = f.read().split("\n")
        f.close()
    except:
        print("Cannot read file.") 
        return
        
    for element in input:

        temp = element.split(",")
        if(temp[0] != '' and temp [1] != '' and temp[2] != ''):
            signal.append( int(temp[0]))
            noise.append( int(temp[1]))
            dist.append( int(temp[2]))
    return output

        

getData(file)

signal1 = []
signal2 = []
noise1 = []
noise2 = []


for i in range(len(dist)):
    if(noise[i] > 36): #overflow of pixels in image makes these unrelyable
        continue
    if(dist[i] == 1):
        signal1.append(signal[i])
        noise1.append(noise[i])
    else:
        signal2.append(signal[i])
        noise2.append(noise[i])
        
        

plt.plot(signal1, noise1, "bo")
plt.plot(signal2, noise2, "ro")

plt.xlabel('$\Delta$I')
plt.ylabel('Noise')  

plt.savefig('detectableParameters.pdf')


  
        