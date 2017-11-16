"""
goal of this example is to simply work with hdf files and their properties

"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Open data file in read mode
dir = './../tempfiles/'
filepath = dir+'data.hdf5'
f = h5py.File(filepath, 'r') #open existing file in read-only mode
ks = f.keys()
k = next(iter(ks))
dset = f[k]['timelapse']
dsize = dset.shape
print(dsize[1])
# data = np.array(dset[:,:,1])
# plt.imshow(data)
# plt.show()

# List all groups in it
# for index, key in enumerate(ks[:2]):
#     print (index, key)
#     data = np.array(f[key].values())
#     print(data.shape())
