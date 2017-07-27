# -*- coding: utf-8 -*-
"""
The "monitor" module recovers the track coordianates of multiple particles 
as they go through the field of view. It assumes uni-directional drift towards 
z = +inf

v0.2, 21 sep.2015, 
@author: SaFa {S.Faez@uu.nl}
"""
#libraries for working with arrays
import gc
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import signal

#libraries for plotting output
import matplotlib as mpl
import matplotlib.pyplot as plt

def centroid(segment, origin):
    """calculates the total mass and the centroid position 
        
    Parameters
    ----------
    segment: input data, 1d array of intensities  
    origin: original index in the mother array
    
    Returns
    -------
    cen: absolute center (measured relative to the origin of the mother array) 
    mass: total mass
    width: first cumulant around center
    """
    scale = np.arange(0,len(segment))
    mass = sum(segment)
    cen = np.dot(scale, segment)/mass
    width = np.sqrt(np.dot((scale-cen)**2, segment))
    cen += origin
    return [cen, mass, width]
    
#%% importing data and setting some ranges and sizes
gc.enable()
filedir = 'X:\Sanli\Data_UI\\150608_nanoMillikan_GNP60\Data_converted\F660_500us\\'
filename = 'F660_2_0p5ms_red_gnp60_AC100Hz00059.dat'
myfile = filedir + filename    
df = pd.read_csv(myfile, index_col=0)
[pix,frames] = df.shape
rawdata = df.values #for the waterfall data, I 
myplot = plt.pcolor(rawdata)
del df
print('File imported!')
#%% cleanup noise
row = rawdata[:,1]
dark = np.median(row)
rawdata = rawdata - dark
#%% finding initial peaks in the first few lines; iterates as long as some peaks are found
psize = 25
min_sig = 40000
widths = np.arange(psize-5,psize+5) # peak-widths that are acceptable for peak finder
peaks_indx = []; cur_t = 0
while (len(peaks_indx)==0):  #scanning frame by frame until finds at least one peak
    cur_line = rawdata[:,cur_t]
    cur_line[cur_line<min_sig] = 0
    peaks_indx = signal.find_peaks_cwt(cur_line, widths, min_snr=3)
    cur_t += 1
print('Particles found at', peaks_indx)
#%% turning peak indices into particle positions and initializing the tracks data structure
columns = ['tag', 't', 'z', 'mass', 'width']    
tracks = DataFrame([[0,0,0,0,0]], columns=columns)
cur_line = rawdata[:,cur_t-1]
p_tag = len(peaks_indx) #largest particle tag for the entries of peak_indx
for pp in np.arange(p_tag,0,-1):
    p = peaks_indx[pp-1]    
    if p<psize:
        seg = cur_line[0:p+psize]
        ori = 0
    elif p>(pix-psize-1):
        seg = cur_line[p-psize:pix-1]
        ori = p-psize
    else:
        seg = cur_line[p-psize:p+psize]
        ori = p-psize
    peak_attr = centroid(seg, ori)
    p_tag = len(peaks_indx)
    new_row = DataFrame([np.concatenate(([p_tag-pp+1, cur_t-1], peak_attr))], columns=columns)
    tracks = tracks.append(new_row, ignore_index=True)
#%%tracing the peaks and checking for new particle arrivals at x=0
"""
The following simplification are made for the following code
- drift is not more than pdrift per frame
- when tracks overlap, the rest of the track is common for two particle
  until the exit the field of view
- drift is from lower to higher index in each line  

To be considered in the future
- colocalization for particles that come too close
- discontinuation of tracks for particles that suddenly disappear
- 
last updated in v0.2, 21 September 2015
"""
pdrift = 1 #estimated particle drift per frame in pixels
for t in np.arange(cur_t,frames):
    cur_line = rawdata[:,t]
    for pp in np.arange(len(peaks_indx),0,-1):
        p = peaks_indx[pp-1]    
        if p<psize:
            seg = cur_line[0:p+pdrift]
            p = np.argmax(seg)
            if p<psize:
                seg = cur_line[0:p+psize]
                ori = 0
            else:
                seg = cur_line[p-psize:p+psize]
                ori = p-psize
            peak_attr = centroid(seg, ori)
            new_row = DataFrame([np.concatenate(([p_tag-pp+1, t], peak_attr))], columns=columns)
            temp = tracks.append(new_row, ignore_index=True)
            del tracks # This is to free up memory which gets stuff by pandas append command
            tracks = temp
            del temp
            peaks_indx[pp-1] = p
        elif p>(pix-pdrift-1):
            peaks_indx = np.delete(peaks_indx,pp-1) #terminates the track if it is too close to the edge
        else:
            seg = cur_line[p-psize:p+psize+pdrift]
            p = p-psize+np.argmax(seg)
            if p>(pix-psize-1):
                peaks_indx = np.delete(peaks_indx,pp-1)
            else:
                seg = cur_line[p-psize:p+psize]
                ori = p-psize    
            peak_attr = centroid(seg, ori)
            new_row = DataFrame([np.concatenate(([p_tag-pp+1, t], peak_attr))], columns=columns)
            tracks = tracks.append(new_row, ignore_index=True)
            peaks_indx[pp-1] = p
    if (len(peaks_indx)==0 or min(peaks_indx)>pdrift+psize):
        seg = cur_line[0:pdrift+psize]
        if (max(seg)>(2*min_sig)):
            newpeak = np.array([np.argmax(seg)])
            peaks_indx = np.concatenate((newpeak, peaks_indx))
            p_tag += 1
    if (np.mod(t,1000) == 0):
        print(t, 'frames analyzed.')        
            
#%%ploting the extracted tracks (just for checking) 
            
#%%saving data
tracks.to_csv('tracks.dat', float_format='%.2f') #uncomment to save generated tracks to file                