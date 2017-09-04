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
filepath = dir + 'wf_10nmAuwDNA1_f2k.npy' #filepath for compressed data in form of a waterfall
wf = np.load(filepath)
# removebg = rsbg()
# wf_clean = removebg.removeWaterfallBG(wf)

newfile =  dir + 'wf_10nmAuwDNA1_f2k_nbg.npy' #filepath for background corrected waterfall
# np.save(newfile, wf_clean)
wf_clean = np.load(newfile)

#plotting a section of the data
sect = wf_clean[1250:1600,1300:1800]
frame, fov = np.shape(sect)
# applying the noise reduction spatial filter
fft_sect = np.fft.rfft2(sect, axes=(-2,))
psize = 9 # nominal particle size in pixels
max_freq = int(fov / psize)
fft_sect[max_freq:, :] = 0
sect_smooth = abs(np.fft.irfft2(fft_sect, axes=(-2,)))
plt.imshow(sect, aspect='auto', interpolation="nearest", vmin = 100, vmax = 3000, extent=(0,frame*0.005,0,fov*0.180), cmap=plt.get_cmap('cool'))
plt.ylabel('position [micrometer]')
plt.xlabel('time [second]')



# plotting data and clean data
# plt.subplot(3, 2, 1)
# plt.imshow(wf, aspect='auto', cmap=plt.get_cmap('cool'))
# plt.title('Compressed waterfall')
# plt.ylabel('z/pixels')
# #plt.xlabel('frame number')
# print('Raw: Max %s, Median %s' %(np.max(wf), np.median(wf)))
#
# plt.subplot(3, 2, 2)
# plt.imshow(wf_clean, aspect='auto', cmap=plt.get_cmap('cool'))
# plt.title('Signal - median background')
# #plt.xlabel('frame number')
# print('Signal: Max %s, Background %s' %(np.max(wf_clean), np.mean(wf_clean)))
#
# plt.subplot(3, 2, 3)
# plt.imshow(np.log10(0.1+wf_clean), aspect='auto', cmap=plt.get_cmap('hot'))
# plt.title('Log(Signal - median background)')
# plt.ylabel('z/pixels')
# #plt.xlabel('frame number')
# print('Signal: Max %s, Background %s' %(np.max(wf_clean), np.mean(wf_clean)))
#
# # applying the noise reduction spatial filter
# fft_wf = np.fft.rfft2(wf_clean, axes=(-2,))
# psize = 10 # nominal particle size in pixels
# max_freq = int(fov / psize)
# fft_wf[max_freq:, :] = 0
# wf_smooth = abs(np.fft.irfft2(fft_wf, axes=(-2,)))
#
# plt.subplot(3, 2, 4)
# plt.imshow(np.log10(0.1+wf_smooth), aspect='auto', cmap=plt.get_cmap('hot'))
# plt.title('Log(Signal), fourier filtering')
# #plt.xlabel('frame number')
# print('Signal: Max %s, Background %s' %(np.max(wf_smooth), np.mean(wf_smooth)))
#
# fnum = 1410
# plt.subplot(3, 2, 5)
# linecut = wf_clean[:,fnum]
# plt.plot(linecut)
# plt.title('Signal - median bg')
# plt.xlabel('z/pixels')
# print('Line: Max %s, Mean %s, Median %s' %(np.max(linecut), np.mean(linecut), np.median(linecut)))
#
# plt.subplot(3, 2, 6)
# linecut = wf_smooth[:,fnum]
# plt.plot(linecut)
# plt.title('Signal - median bg')
# plt.xlabel('z/pixels')
# print('Line: Max %s, Mean %s, Median %s' %(np.max(linecut), np.mean(linecut), np.median(linecut)))
#
#


plt.show()


# # identifying the tracks in the waterfall
# trackbot = TracksZ(psize=9, drift=1, snr = 20, noiselvl= 1)
# # iniloc = trackbot.locateInitialPosition(wf)
# identified_tracks = trackbot.collectTracks(wf)
# track1 = identified_tracks[identified_tracks[:,0]==1]
# print(track1.shape)
# ## plotting waterfall, original tracks, and identifies tracks to compare
# plt.subplot(1, 3, 1)
# plt.imshow(wf, aspect='auto', cmap=plt.get_cmap('cool'))
# plt.title('Waterfall')
# plt.ylabel('z/pixels')
# plt.xlabel('frame number')
# print('Raw: Max %s, Median %s' %(np.max(wf), np.median(wf)))
#
# plt.subplot(1, 3, 2)
# plt.plot(simulated_track1[:,1],simulated_track1[:,3])
#
# plt.subplot(1, 3, 3)
# plt.plot(track1[:,1],track1[:,3])
