"""
    CETgraph.example.py
    ==================================
    This is an example to show how to use this package
    quick description:
        - simulate some waterfall
        - analyze this waterfall using the tracking routines
        - [not yet implemented] show results of tracking (and compare with actual parameters)

    .. lastedit:: 27/7/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""

from tracking.simulate import Waterfall
from tracking.identify import Locations


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

setup = Waterfall(size=200, signal=50)
wf=setup.genwf()
plt.imshow(wf)
plt.show()