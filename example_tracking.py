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
# simulating a new waterfall
setup = Waterfall(fov=200, numpar=1, difcon = 3, signal = 50, noise = 1, psize = 8, drift = 0.5)
wf = setup.genwf()
simulated_tracks = setup.tracks
simulated_track1 = simulated_tracks[simulated_tracks[:,0]==1]
# identifying the tracks in the waterfall
trackbot = TracksZ(psize=9, drift=1, snr = 20, noiselvl= 1)
# iniloc = trackbot.locateInitialPosition(wf)
identified_tracks = trackbot.collectTracks(wf)
track1 = identified_tracks[identified_tracks[:,0]==1]
print(track1.shape)
## plotting waterfall, original tracks, and identifies tracks to compare
plt.subplot(1, 3, 1)
plt.imshow(wf, aspect='auto', cmap=plt.get_cmap('cool'))
plt.title('Waterfall')
plt.ylabel('z/pixels')
plt.xlabel('frame number')
print('Raw: Max %s, Median %s' %(np.max(wf), np.median(wf)))

plt.subplot(1, 3, 2)
plt.plot(simulated_track1[:,1],simulated_track1[:,3])

plt.subplot(1, 3, 3)
plt.plot(track1[:,1],track1[:,3])


plt.show()