"""
exmaple for tracking diffusion+drifting particles of relatively high SNR
    .. lastedit:: 9/8/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""


from tracking.cleanup import RemoveStaticBackground as rsbg
from tracking.simulate import Waterfall
from tracking.identify import TracksZ
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# dir = 'c:/tmp/tests/CETgraph/'
# filepath = dir + 'wf_test.npy' #filepath for compressed data in form of a waterfall generated using simulate.py
#
# wf = np.load(filepath)
setup = Waterfall(fov=200, numpar=1, difcon = 3, signal = 50, noise = 1, psize = 8, drift = 1)
wf = setup.genwf()
print(wf.shape)
trackbot = TracksZ(psize=9, drift=1, snr = 20, noiselvl= 1)
# iniloc = trackbot.locateInitialPosition(wf)
alltracks = trackbot.collectTracks(wf)
print(alltracks.shape)
track1 = alltracks[alltracks[:,0]==1]
print(track1.shape)
## plotting data and clean data
plt.subplot(1, 2, 1)
plt.imshow(wf, aspect='auto', cmap=plt.get_cmap('cool'))
plt.title('Waterfall')
plt.ylabel('z/pixels')
plt.xlabel('frame number')
print('Raw: Max %s, Median %s' %(np.max(wf), np.median(wf)))

plt.subplot(1, 2, 2)
plt.plot(track1[:,1],track1[:,3])

plt.show()