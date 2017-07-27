"""
    CETgraph.tracking.lib.calc.py
    ==================================
    General functions used in various routines

    .. lastedit:: 27/7/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np

def centroid(segment, origin):
    """calculates the total mass and the centroid position

    Parameters
    ----------
    segment: input data, 1d array of intensities
    origin: original index in the mother array

    Returns
    -------
    cen: absolute center (measured relative to the origin of the mother array)
    mass: total mass
    width: first cumulant around center
    """
    scale = np.arange(0, len(segment))
    mass = sum(segment)
    cen = np.dot(scale, segment) / mass
    width = np.sqrt(np.dot((scale - cen) ** 2, segment))
    cen += origin
    return [cen, mass, width]