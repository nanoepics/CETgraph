"""
    CETgraph.example.py
    ==================================
    This is an example of analysis code to show how to use this package
    The content depends on the actual tests on the code.
    To get some more inspiration, check also other example files and files in .tests
    generally, a well prepared analysis code only handles the input and output data. All other generic calculations
    should be done via the methods in other CETgraph modules and libraries.
    .. lastedit:: 27/7/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""

from tracking.compress import Waterfall

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

# Open data file in read mode
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
wf = setup.compressHDF(f, nframes=240, cenline=230, linewidth=20)
plt.imshow(wf)
plt.show()