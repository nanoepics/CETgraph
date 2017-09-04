"""
    CETgraph.example_compress_sequence.py
    ==================================
    This is an example for reading out a sequence of data from hdf5 file and compressing it into raw waterfall data.
    .. lastedit:: 7/8/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""

from tracking.compress import Waterfall

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

# Open measurement file in read-only mode
dir = 'D:/Matej/2017-07-05/'
filepath = dir+'10nmAuwDNA1.hdf5'
f = h5py.File(filepath, 'r') #open existing file in read-only mode

# following lines are only necessary to find out the index of fiber position in image (cenline) and linewidth of the bright stripe
# k = next(iter(f.keys()))
# dset = f[k]['timelapse']
# snap = np.array(dset[:,:,1])
# plt.imshow(snap)
# plt.show()
# # #


setup = Waterfall()
wf = setup.compressHDF(f, nframes=10570, cenline=0, linewidth=0)

# to save data in a numpy array, it is good enough for 2d arrays
out_dir = 'D:/Matej/170830processed/'
out_file = 'wf_10nmAuwDNA1_f10k'
print(out_dir+out_file)
np.save(out_dir+out_file, wf)

plt.imshow(wf)
print(wf.shape)
plt.show()