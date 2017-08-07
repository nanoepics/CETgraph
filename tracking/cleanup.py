"""
    CETgraph.tracking.compress.py
    ==================================
    These routines remove the static background from a sequence of images with various methods

    .. lastedit:: 7/8/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np
from .lib import calc

class RemoveStaticBackground:
    """
    :param
    :return:
    """
    def __init__(self):
        self.nframes = 0
        self.fov = 0
        self.method = 'median'

    def removeWaterfallBG(self, data, method = 'median', delay = 30):
        """

        :param data: raw data in form of a waterfall
        :param method: bg correction; so far implemented 'median' or 'moving'
        :param delay: for the method 'moving' the delay after which is taken as bg
        :return:
        """
        [fov, nframes] = data.shape
        self.fov = fov
        self.nframes = nframes
        wf = 0*data
        if method == 'median':
            bg = np.median(data, axis=1)
            for i in range(nframes):
                wf[:,i] = data[:,i] - bg
                wf[wf<0] = 0
        elif method == 'moving':
            for i in range(nframes):
                bgi = np.mod((i+delay), nframes)
                wf[:, i] = data[:, i] - data[:, bgi]
                wf[wf < 0] = 0

        return wf

    def removeNoise(self):
        print("removeNoise to be implemented...")