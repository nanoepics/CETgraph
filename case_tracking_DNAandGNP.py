"""
exmaple for tracking diffusion+drifting particles of relatively high SNR
    .. lastedit:: 9/8/2017
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
sect = np.flipud(wf[:,0:200])
frame, fov = np.shape(sect)
# applying the noise reduction spatial filter
fft_sect = np.fft.rfft2(sect, axes=(-2,))
psize = 9 # nominal particle size in pixels
max_freq = int(fov / psize)
fft_sect[max_freq:, :] = 0
sect_smooth = abs(np.fft.irfft2(fft_sect, axes=(-2,)))



# identifying the tracks in the waterfall
trackbot = TracksZ(psize=9, drift=1, snr = 1.5, noiselvl= 100)
# iniloc = trackbot.locateInitialPosition(wf)
identified_tracks = trackbot.collectTracks(sect_smooth, loca=[229, 1138])
print(identified_tracks.shape)
track1 = identified_tracks[identified_tracks[:,0]==1]

## plotting waterfall, original tracks, and identifies tracks to compare
#plt.subplot(1, 2, 1)
plt.imshow(sect, aspect='auto', interpolation="nearest", vmin = 100, vmax = 3000, origin = 'lower', cmap=plt.get_cmap('cool'))
plt.ylabel('position [micrometer]')
plt.xlabel('time [second]')

#plt.subplot(1, 2, 2)
plt.scatter(identified_tracks[:,1],identified_tracks[:,3], marker=',')


plt.show()

