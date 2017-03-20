# -*- coding: utf-8 -*-
"""
"code_template.py" is a minimal collection of good practices in programming 
with Python. It follows the Python style guide 
https://www.python.org/doc/essays/styleguide/ 

You can use this file as a template for your own code. 

v0.1, 18 feb. 2015, 
@author: SaFa {S.Faez@uu.nl}
"""

#libraries for working with arrays
import numpy as np
from pandas import DataFrame, Series
#libraries for plotting output
import matplotlib as mpl
import matplotlib.pyplot as plt

#np.random.seed(0)

#----------------------------------------------------------
# Constants
#----------------------------------------------------------

#----------------------------------------------------------
# Classes and Functions
#----------------------------------------------------------

def layers(n, m):
    """
    Function returns random Gaussian mixtures.
            
    Parameters
    ----------
    n: number of generated curves 
    m: size of window for smoothing the curve
    
    Returns
    -------
    a: numpy array of size (m, n) representing the generated values
    
    Notes
    -----
    nothing to add
    """
    def bump(a):
        x = 1 / (.1 + np.random.random())
        y = 2 * np.random.random() - .5
        z = 10 / (.1 + np.random.random())
        for i in range(m):
            w = (i / float(m) - y) * z
            a[i] += x * np.exp(-w * w)
    a = np.zeros((m, n))
    for i in range(n):
        for j in range(5):
            bump(a[:, i])
    return a

#----------------------------------------------------------
# Main Program Entry
#----------------------------------------------------------
d = layers(3, 100)
plt.subplots()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Some colorful bumpy curves")
plt.stackplot(range(100), d.T, baseline='wiggle')
plt.show()
