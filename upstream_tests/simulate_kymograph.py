"""
    CETgraph.tests.simulate_kymograph.py
    ==================================
    This simple test, generates a synthetic kymograph using the .simulate class.
    generally, a well prepared analysis code only handles the input and output data. All other generic calculations
    should be done via the methods in other CETgraph modules and libraries.
    .. lastedit:: 27/7/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""

from tracking.simulate import Kymograph

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

setup = Kymograph(fov=300, numpar=3, difcon=1, nframes=500, signal=50, drift = 0.8)
kg = setup.genKymograph()
plt.imshow(kg)
plt.title('Synthetic Kymograph')
plt.ylabel('z/pixels')
plt.xlabel('frame number')
plt.show()

# to save data in a numpy array, it is good enough for 2d arrays
out_dir = './../tempfiles/'
out_file = 'wf_test'
print(out_dir+out_file)
np.save(out_dir + out_file, kg)