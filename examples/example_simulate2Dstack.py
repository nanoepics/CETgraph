"""
    CETgraph.examples.simulate2Dstack.py
    ==================================
    This example, generates a synthetic stack of 2d frames using the .simulate class.
    generally, a well prepared analysis code only handles the input and output data. All other generic calculations
    should be done via the methods in other CETgraph modules and libraries.
    .. lastedit:: 1/3/2018
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""

from tracking.simulate import SingleFrame

import numpy as np
import matplotlib.pyplot as plt


setup = SingleFrame(fov=[300,150], numpar=5, difcon=1, signal=20, noise = 1, psize = 8)
stack = setup.genStack(20)
print(np.shape(stack))
plt.imshow(stack[:,:,0])
plt.show()
