# -*- coding: utf-8 -*-
"""
The "curvefit_slow" module uses the tracks found by "monitor" to find the time 
constants of the actuated drift. It is best for low-frequency driving 
oscillations because it find the turning points by peak-finder. For rapidly
oscillating driving field, use the "find_mobility" module

v0.1, 27 jan. 2015, 
@author: SaFa {S.Faez@uu.nl}
"""
#libraries for working with arrays
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import signal

#libraries for plotting output
import matplotlib as mpl
import matplotlib.pyplot as plt
def dot(curve, window=3):
    """calculates the derivative of the curve assuming n_points to
    overcome noise
        
    Parameters
    ----------
    curve: input data, 1d array (of usually positions) 
    n_points: size of window for smoothing the curve
    Returns
    -------
    out: second derivative 
    """
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(curve, weights, 'valid')
    dcurve = np.diff(sma)
    out = dcurve
    return out
#%%Import track information and set time and legth scales
myfile = 'tracks100Hz.dat' 
df = pd.read_csv(myfile, index_col=0)
#plot1 = plt.plot(df.z)
dt = 0.5 #time step in milliseconds
dx = 2 * 0.0644 #pixel size in micrometers
#%%Finds the segments by looking for the turning points
positions = df.z.values
velocities = dot(positions,1)*dt/dx
accelarations = np.diff(velocities)*dt
widths = np.arange(15,25) # peak-widths that are acceptable for peak finder
turn_indx = signal.find_peaks_cwt(np.abs(velocities), widths, min_snr=4)
segments = len(turn_indx)
turn_indx = np.append(turn_indx,velocities.size)

#%%Finds the segments by extracting the periodicity
plt.plot(positions[1:200]-500)
plt.plot(velocities[1:200], marker='o')
#%%Fits exponential to each section of the curve and gathers stat
avgseg_len = min(np.diff(turn_indx))-10
avgseg = np.zeros(avgseg_len)
for i in np.arange(1,segments):
    segment = np.abs(velocities[turn_indx[i]-2:turn_indx[i+1]-5])
    avgseg = avgseg + segment[0:avgseg_len]/segments
    #plt.plot(segment)
#%%Plotting the average curve
plt.figure(1,figsize=(6,12),dpi=300)
plt.subplot(311)
plt.plot(avgseg)
plt.ylabel('velocity(um/s)'); plt.xlabel('time(ms)')
plt.yscale('log'); plt.xscale('log');plt.xlim((0,400)); plt.ylim((1,40))

plt.subplot(312)
plt.plot(avgseg)
plt.ylabel('velocity(um/s)'); plt.xlabel('time(ms)')
plt.yscale('log'); plt.xscale('linear');plt.xlim((0,400)); plt.ylim((1,40))

plt.subplot(313)
plt.plot(avgseg)
plt.ylabel('velocity(um/s)'); plt.xlabel('time(ms)')
plt.yscale('linear'); plt.xscale('log');plt.xlim((0,400)); plt.ylim((1,40))

plt.savefig('plot.png', dpi=300)
plt.show()

    
 
    
    