"""
How to read data from HDF files and find features using trackpy
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import trackpy as tp

dir = 'F:/Data/2018-02-13/'
filepath = dir+'100xDNA2.hdf5'
#filename = 'example_data.hdf5'
f = h5py.File(filepath, 'r') # open existing file in read-only mode

k = next(iter(f.keys())) # getting the first key
dset = f[k] ['timelapse'] # corresponding dataset
print(dset.shape)

movie = np.array(dset[:,:,:400]).T # stack of images of the dset
snap = movie[40,:,:] # read out a single frame from the stack
#plt.imshow(snap)
#plt.show()

#for i in range(40, 41):
#    snap = movie[i,:,:]  # testing parameters like diameter and minmass
#    s = tp.locate(snap, diameter=25, minmass=3000)  # locating features in a single frame
#    plt.figure()
#    tp.annotate(s, snap)

m = tp.batch(movie[40:110], diameter=19, minmass=3000)  # locating feature in the stack
trajectoryRaw = tp.link_df(m, 75, memory=3)  # linking located features
trajectoryFiltered = tp.filter_stubs(trajectoryRaw, 10)  # filtering spurious trajectories

fig, ax = plt.subplots()
fig.tight_layout() 
ax = tp.mass_size(trajectoryFiltered.groupby('particle').mean()) # plot average size and mass of each linked trajectories

fig, ax = plt.subplots()
fig.tight_layout() 
ax = tp.plot_traj(trajectoryFiltered)  # plot the trajectories

d = tp.compute_drift(trajectoryFiltered, 15)
d.plot() # plot the drift

trajectory = tp.subtract_drift(trajectoryFiltered.copy(), d) # final trajectory without diffusion
fig, ax = plt.subplots()
fig.tight_layout() 
ax = tp.plot_traj(trajectory) # plot all the trajectories

im = tp.imsd(trajectory, 0.104, 100) # calculation of the mean squared displacement vs lag time
fig, ax = plt.subplots() # plotting MSD vs lag time 
fig.tight_layout() 
ax.plot(im.index, im, 'k-', alpha=0.99)  # black lines, semitransparent
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
ax.set_xscale('log')
ax.set_yscale('log')

imNarray = im.reset_index().values # convert it into numpy array
pfit = np.polyfit(imNarray[:10, 0], imNarray[:10, 1], 1) # performs linear fit for the first 10 elements
diff_const = pfit[0]/2 # diffusion constant in um^2/s
diameter = 1000 * 2 * 4.04 / (6 * 3.1415 * 1.002 * diff_const) # diameter in nanometer assuming D in um^2/s
print(diameter)







