"""
exmaple for tracking diffusion+drifting particles of relatively dilute sample
    .. lastedit:: 5/9/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""


from tracking.cleanup import RemoveStaticBackground as rsbg
#from tracking.simulate import Waterfall
from tracking.analyze import DiffusionZ as difz
import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rc('font', family='serif', size=24)

dir = 'C:/Users/ebrah002/Documents/Data_UI/Matej/170830processed/'
tracksfile =  dir + 'tracks_10nmAuwDNA1_f2k.npy' #filepath for background corrected waterfall
identified_tracks = np.load(tracksfile)

# choosing one of the tracks to get the diffusion constant

msdbot = difz(tstep = 0.005, pix = 0.134, nsteps = 7, show_plots = True)
track1 = identified_tracks[identified_tracks[:,0]==1]
#plt.plot(track1[:,3])
#plt.show()

diff_const = msdbot.findDiffConstant(track1)
print(diff_const)

# extracting effective from diffusion constant assuming spherical particle r = kT/6pi.D.eta
# eta = 1.002 e-3 Pa.s at T=293 K, kT = 4.04 e-21,
diameter = 1000 * 2 * 4.04 / (6 * 3.1415 * 1.002 * diff_const)  # diameter in nanometer assuming D in um^2/s
print(diameter)