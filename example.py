"""
    CETgraph.example.py
    ==================================
    This is an example of analysis code to show how to use this package
    quick description:
        - simulate some waterfall
        - analyze this waterfall using the tracking routines
        - [not yet implemented] show results of tracking (and compare with actual parameters)

    generally, a well prepared analysis code only handles the input and output data. All other generic calculations
    should be done via the methods in other CETgraph modules and libraries.
    .. lastedit:: 27/7/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""

from tracking.simulate import Waterfall
from tracking.identify import Locations


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

setup = Waterfall(fov=200, n=4, signal=50)
wf = setup.genwf()
plt.imshow(wf)
plt.show()