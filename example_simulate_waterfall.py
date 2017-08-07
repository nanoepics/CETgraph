"""
simple exmaple of how to use the classes in CETgraph to generate a synthetic waterfal
    .. lastedit:: 27/7/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""

from tracking.simulate import Waterfall
from tracking.identify import Locations


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

setup = Waterfall(fov=200, numpar=4, signal=50)
wf = setup.genwf()
plt.imshow(wf)
plt.show()