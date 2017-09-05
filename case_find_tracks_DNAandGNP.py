"""
example for tracking diffusion+drifting particles of relatively dilute sample
    .. lastedit:: 5/9/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""


from tracking.cleanup import RemoveStaticBackground as rsbg
#from tracking.simulate import Waterfall
from tracking.identify import TracksZ
import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rc('font', family='serif', size=24)

dir = 'C:/Users/ebrah002/Documents/Data_UI/Matej/170830processed/'
newfile =  dir + 'wf_10nmAuwDNA1_f2k_nbg.npy' #filepath for background corrected waterfall
wf = np.load(newfile)

#plotting a section of the data
sect = np.flipud(wf)
frame, fov = np.shape(sect)
# applying the noise reduction spatial filter
fft_sect = np.fft.rfft2(sect, axes=(-2,))
psize = 9 # nominal particle size in pixels
max_freq = int(fov / psize)
fft_sect[max_freq:, :] = 0
sect_smooth = abs(np.fft.irfft2(fft_sect, axes=(-2,)))

# identifying the tracks in the waterfall
trackbot = TracksZ(psize=10, step=6, drift=1, snr = 1.5, noiselvl= 400)
#iniloc = trackbot.locateInitialPosition(sect_smooth)
iniloc = [49, 229, 1137]
identified_tracks = trackbot.followLocalMax(sect, iniloc)
print(identified_tracks.shape)
track1 = identified_tracks[identified_tracks[:,0]==1]

## plotting waterfall, original tracks, and identifies tracks to compare
#plt.subplot(1, 2, 1)
plt.imshow(sect, aspect='auto', interpolation="nearest", vmin = 100, vmax = 3000, origin = 'lower', cmap=plt.get_cmap('cool'))
plt.ylabel('position [pixels]')
plt.xlabel('time [frames]')

#plt.subplot(1, 2, 2)
plt.scatter(identified_tracks[:,1],identified_tracks[:,3], marker=',')

outfile = dir + 'tracks_10nmAuwDNA1_f2k.npy'
np.save(outfile, identified_tracks)
plt.show()

