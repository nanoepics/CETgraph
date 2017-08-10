# -*- coding: utf-8 -*-
"""
The "analyze" module processes the tracks as extracted by the "monitor" module. 
Most functions are either directly taken from Trackpy (github.com/soft-matter/trackpy)
or adapted to serve the main functions of nanoCapillary tracking

Suggested steps for analysis
1- choose path(s)
1'- optional: check pixel bias
2- calculate drift (average drift over all tagged  particles)
3- plot MSD
3'- optional: plot step size distribution
4- plot D vs avg intensity
5- normalize D and intensity
6- present statistics of particle size

v0.1, 21 sep.2015, 
@author: SaFa {S.Faez@uu.nl}
"""
#libraries for working with arrays
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

#libraries for plotting output
import matplotlib as mpl
import matplotlib.pyplot as plt

#%% importing data and setting some ranges and sizes
uz = 0.129 #length scale in micrometer/pixel (considering 2x2 binning)
ut = 0.01 #time scale in seconds/frame 
df = pd.read_csv('tracks.dat', index_col=0)

