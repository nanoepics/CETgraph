import numpy as np

a = np.array([2,3])
b = np.array([a,a])
c = np.concatenate((b,[a]), axis=0)
print(c)