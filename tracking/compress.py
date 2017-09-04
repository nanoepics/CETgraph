"""
    CETgraph.tracking.compress.py
    ==================================
    These routines reduce the dimension of measured data to the minimun necessary for the analysis for example from a data cube to a waterfall (in the chromotography communities known as the chemograph)

    .. lastedit:: 7/8/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np
#from .lib import calc
import h5py

class Waterfall:
    """Generates z-position vs time from a measurement sequence by averaging over the axis perpendicular to the fiber
    input:
    datafile location of the hdf5 file  (in __future__ a general imagecube)


    :param:
    nframes: total number of non-empty frames in the sequence
    fov: field of veiw (after compression) in pixels

    for __future__ these parameters can be estimated in generation of the waterfall to make class consistent with the simulate.Waterfall method
    ----------
    numpar: estimated number of particles in the field of view
    difcon: diffusion constant (identical particles) [pixel^2/frame]
    psize: particle size in pixels
    signal: brightness of each particle
    noise: background random noise
    drift: average drift velosity [pixel/frame]
    ----------

    """

    def __init__(self):
        self.nframes = 0
        self.fov = 0


    def compressHDF(self, file, key = [], nframes = 0, cenline = 0, linewidth = 0):
        """
        :param
        file: HDF5 file containing one or multiple sequences of images
        key: key to specific group in the hdf5 file. If empty, method will take the last sequence in the file
        nframes: desired number of frame in the waterfall. If 0, method will set it to maximum number of non-empty frames
        cenline: the line around with data should be averaged
        linewidth: half image width around cenline that should be averaged
        :returns
        wf: float numpy array of intensity(position,time)
    """
        if not key:
            ks = file.keys()
            k = next(iter(ks))
        else:
            k = key
        dset = file[k]['timelapse']    # for the moment, either the key input is properly set assumes the hdf5 file  starts with an actual group of type 'timelapse' and not a 'snap'. should be taken care of in __future__
        dsize = dset.shape
        # setting the main parameters of averaging if they are not explicitly mentioned
        if (nframes == 0 or nframes > dsize[2]):
            nframes = dsize[2]
        self.nframes = nframes
        if cenline == 0:
            cenline = int(dsize[1]/2)
        if (linewidth == 0 or linewidth > dsize[1]):
            linewidth = int(dsize[1]/2)
        fov = dsize[0]
        self.fov = fov
        #print(fov)
        wf = np.zeros((fov, nframes))
        for i in range(nframes):
            image = np.array(dset[:, cenline-linewidth:cenline+linewidth, i])
            wf[:,i] = np.sum(image, axis=1)
            print(i) # counter for showing progress in compressing images into lines

        return wf



