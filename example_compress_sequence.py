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
dir = 'C:/tmp/data/Tracking 5nm particle/'
filepath = dir+'BlueParticle5nm1.hdf5'
f = h5py.File(filepath, 'r') #open existing file in read-only mode

# following lines are only necessary to find out the index of fiber position in image (cenline) and linewidth of the bright stripe
# k = next(iter(f.keys()))
# dset = f[k]['timelapse']
# snap = np.array(dset[:,:,1])
# plt.imshow(snap)
# plt.show()

setup = Waterfall()
wf = setup.compressHDF(f, nframes=400, cenline=230, linewidth=20)
plt.imshow(wf)
#print(wf.shape)
plt.show()

# to save data in a numpy array, it is good enough for 2d arrays
out_dir = 'c:/tmp/tests/CETgraph/'
out_file = 'wf5nm'
print(out_dir+out_file)
np.save(out_dir+out_file, wf)