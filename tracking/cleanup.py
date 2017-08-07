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
    def __init__(self, data, method = 'median', delay = 30):