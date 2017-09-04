"""
simple exmaple of how to correct backgroun and track particles on waterfall data
    .. lastedit:: 27/7/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""

from tracking.cleanup import RemoveStaticBackground as rsbg
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

dir = 'c:/tmp/tests/CETgraph/'
filepath = dir + 'wf_test.npy' #filepath for compressed data in form of a waterfall

wf = np.load(filepath)
wf_clean = rsbg().removeWaterfallBG(wf)
wf_moving = rsbg().removeWaterfallBG(wf, method='moving')
# plotting data and clean data
plt.subplot(1, 3, 1)
plt.imshow(wf, aspect='auto', cmap=plt.get_cmap('inferno'))
plt.title('Compressed waterfall')
plt.ylabel('z/pixels')
plt.xlabel('frame number')
print('Raw: Max %s, Median %s' %(np.max(wf), np.median(wf)))

plt.subplot(1, 3, 2)
plt.imshow(wf_clean, aspect='auto', cmap=plt.get_cmap('cool'))
plt.title('Signal - median bg')
plt.xlabel('frame number')
print('Signal: Max %s, Median %s' %(np.max(wf_clean), np.median(wf_clean)))

plt.subplot(1, 3, 3)
plt.imshow(wf_moving, aspect='auto', cmap=plt.get_cmap('cool'))
plt.title('Signal - moving bg')
plt.xlabel('frame number')
print('Signal: Max %s, Median %s' %(np.max(wf_moving), np.median(wf_moving)))


plt.show()