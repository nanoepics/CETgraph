# -*- coding: utf-8 -*-
"""
The "synthesize" module generates artificial tracks with pre-defined 
physical paramaters as a test-input for the tracking code or other parts 
of the analysis.

v0.1, 7 sep.2015, 
@author: SaFa {S.Faez@uu.nl}
"""

#libraries for working with arrays
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
#libraries for plotting output
import matplotlib as mpl
import matplotlib.pyplot as plt

def gen_tracks_1d(par_dens, diff_cst, v_drift=0,
                  pix=100, frames=50, psize=5, snr=50):
    """Generates z-position vs time for a group of particles as they go with
    normal Brownian motion and drift.
    
    Parameters
    ----------
    par_dens: number of particles in the field of view,
    diff_cst: diffusion constant (considering all identical particles) [pixel^2/frame] 
    v_drift: average drift velosity [pixel/frame]
    pix: field of view in pixels
    frames: total number of synthesized frames
    psize: particle size in pixels
    snr: signal to noise ratio
    
    Returns
    -------
    float numpy array of intensity vs time
    """
    positions = pix*np.random.rand(par_dens)
    tracks = np.zeros((pix,frames))
    taxis = np.arange(frames)
    for p in positions: # generating random-walk assuming dt=1
        steps = np.random.standard_normal(frames) 
        path = p + np.cumsum(steps)*np.sqrt(diff_cst) + v_drift*taxis 
        for t in taxis:
            pt = int(path[t]%pix) # NOTE: generates pixel bias
            tracks[pt, t] += snr
    fft_tracks = np.fft.rfft2(tracks,axes=(-2,))
    max_freq = int(pix/psize)
    fft_tracks[max_freq:,:] = 0
    tracks = abs(np.fft.irfft2(fft_tracks,axes=(-2,)))
    noise = np.random.randn(pix, frames)
    tracks += noise
    return tracks

syn_tracks = gen_tracks_1d(2, 4, 1, 300, 400)
myplot = plt.pcolor(syn_tracks)
df=DataFrame(syn_tracks)
df.to_csv('test_s.dat', float_format='%.3f') #uncomment to save generated tracks to file