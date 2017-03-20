# -*- coding: utf-8 -*-
"""
The "findMobility" module uses the tracks found by "monitor" as input
to find the mobility of the actuated particle. It is best for low-frequency driving 
oscillations because it find the turning points by peak-finder. For rapidly
oscillating driving field, use the "findMobility" module

v0.1, 27 jan. 2015, 
@author: SaFa {S.Faez@uu.nl}
"""
#libraries for working with arrays
import numpy as np
from scipy import fftpack as ff

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
    dcurve = np.diff(curve)
    out = np.convolve(dcurve, weights, 'valid')
    return out
#%%Import track information and set time and legth scales
myfile = 'tracks100Hz.dat' 
df = pd.read_csv(myfile, index_col=0)
#plot1 = plt.plot(df.z)
dt = 0.5 #time step in milliseconds
dx = 2 * 0.0644 #pixel size in micrometers
#%%Finds the segments by looking for the turning points
positions = df.z.values
velocities = dot(positions,5)*dt/dx
plt.plot((velocities[2000:2250]),'*-')
plt.ylabel('Velosity (um/s)'); plt.xlabel('t (ms)')
plt.savefig('plot.png', dpi=300)
#%%Finds the segments by extracting the periodicity and looking at each segment for maximum
vfft = ff.rfft(velocities)
freq = np.argmax(vfft)
segperiod = velocities.size/freq
seglen = np.round(segperiod)
cursor = 0
maxv = np.zeros([freq-2])
for i in np.arange(freq-2):
    seg = np.abs(velocities[cursor:cursor+seglen])
    segmax = np.argmax(seg)
    maxv[i] = seg[segmax]
    cursor = np.floor(max((i*segperiod+segmax-3), 0))
    
plt.plot(maxv[1:400])
#%%Plotting the average curve
hist, bins = np.histogram(maxv/0.042, bins=100)
width = 0.6 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
fig, ax = plt.subplots()
ax.bar(center, hist, align='center', width=width)
plt.ylabel('Frequency'); plt.xlabel('~Q/e')
#fig.yscale('log'); fig.xscale('log'); 

plt.savefig('plot.png', dpi=300)
plt.show()

    
 
    
    