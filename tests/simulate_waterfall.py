"""
    CETgraph.tests.simulate_watefall.py
    ==================================
    This simple test, generates a waterfall using the .simulate class.
    generally, a well prepared analysis code only handles the input and output data. All other generic calculations
    should be done via the methods in other CETgraph modules and libraries.
    .. lastedit:: 27/7/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""

from tracking.simulate import Waterfall

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

setup = Waterfall(fov=200, numpar=4, signal=50)
wf = setup.genwf()
plt.imshow(wf)
plt.show()

# to save data in a numpy array, it is good enough for 2d arrays
out_dir = 'c:/tmp/tests/CETgraph/'
out_file = 'wf_test'
print(out_dir+out_file)
np.save(out_dir+out_file, wf)