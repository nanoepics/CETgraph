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
    def __init__(self, method = 'median', delay = 30):
        self.nframes = 0
        self.bgdelay = delay
        self.fov = 0
        self.bgmethod = method
        self.background = 0

    def removeKymographBG(self, data):
        """

        :param data: raw data in form of a kymograph
        :param method: bg correction; so far implemented 'median' or 'moving'
        :param delay: for the method 'moving' the delay after which is taken as bg
        :return:
        """
        [fov, nframes] = data.shape
        self.fov = fov
        self.nframes = nframes
        delay = self.bgdelay
        method = self.bgmethod
        kg = 0*data
        if method == 'median':
            bg = np.median(data, axis=1)
            for i in range(nframes):
                kg[:,i] = data[:,i] - bg
                kg[kg < 0] = 0
            self.background = np.mean(bg)
        elif method == 'moving':
            for i in range(nframes):
                bgi = np.mod((i+delay), nframes)
                kg[:, i] = data[:, i] - data[:, bgi]
                kg[kg < 0] = 0
            self.background = np.median(bgi)
        return kg

    def removeNoise(self):
        print("removeNoise to be implemented...")