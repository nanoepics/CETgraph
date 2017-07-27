"""
    CETgraph.tracking.identify.py
    ==================================
    Routines that are necessary for locating particles in a frame and tracking them in a series of frames or a waterfall image

    .. lastedit:: 27/7/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np
from .lib import calc

class Locations(object):
    """
    :param
    :return:
    """
    def __init__(self, data, signal = 10, noise = 1, psize = 8):
        self.data = data
        self.signal = signal # brightness for each particle
        self.noise = noise # background noise
        self.psize = psize # half-spread of each particle in the image, currently must be integer
