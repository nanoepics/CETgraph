"""
example for identifying peaks in a kymograph
    .. lastedit:: 9/8/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""


from tracking.simulate import Kymograph
from tracking.identify import TracksZ
import numpy as np
import matplotlib.pyplot as plt

dir = './examples/'
filepath = dir + 'wf_test.npy' #filepath for compressed data in form of a kymograph
wf = np.load(filepath)

# otherwise simulate a new kymograph CETgraph.tracking.simulate
# setup = Kymograph(fov=200, numpar=1, difcon = 3, signal = 50, noise = 1, psize = 8, drift = 0.5)
# wf = setup.genKymograph()
# simulated_tracks = setup.tracks


# identifying the peak positions in the first frames
trackbot = TracksZ(psize=8, drift=1, step=1, snr = 25, noiselvl= 3)
iniloc = trackbot.locateInitialPosition(wf)

# tracking the identified particles
identified_tracks = trackbot.followLocalMax(wf, iniloc, inlet='left')
numtracks = np.int(np.max(identified_tracks[:,0]))
print('%d separate tracks identifies' %(numtracks+1))

## plotting kymograph and identifies tracks to compare

plt.imshow(wf, aspect='auto', cmap=plt.get_cmap('cool'))
plt.title('Identified tracks')
plt.ylabel('z/pixels')
plt.xlabel('frame number')
for i in range(numtracks+1):
    track = identified_tracks[identified_tracks[:,0]==i]
    plt.plot(track[:,1],track[:,3],'.')

print(identified_tracks[-1,:])
plt.show()

filepath = dir + 'tracks_test.npy'
np.save(filepath, identified_tracks)