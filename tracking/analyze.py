# -*- coding: utf-8 -*-
"""

CETgraph.tracking.analyze.py
=================================================
The "analyze" module processes the tracks as extracted by the "identify" module.
Most functions are either directly taken from Trackpy (github.com/soft-matter/trackpy)
or adapted to serve the main functions of nanoCapillary tracking

Suggested steps for analysis
1- choose path(s)
1'- optional: check pixel bias
2- calculate drift (average drift over all tagged  particles)
3- plot MSD
3'- optional: plot step size distribution
4- plot D vs avg intensity
5- normalize D and intensity
6- present statistics of particle size

v0.2, 5 september 2017,
section @author: Sanli Faez {S.Faez@uu.nl}
"""
#libraries for working with arrays
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif', size=24)

class DiffusionZ:
    """
    The "DiffusionZ" module calculated the physical properties of a random walker from its time-position track

    parameters needed to initiate the localization requirements
    :param tstep: temporal difference between two successive frames in seconds
    :param pix: pixel size in micrometers
    :param nsteps: number of delays considered for fitting Mean Square Displacement
    :param show_plots: if True, plot some intermediate plots for quality check


    [not implemented] consistently return success/failure messages regarding locating the particle
    """
    def __init__(self, tstep = 1, pix = 1, nsteps = 12, show_plots = True):
        self.tstep = tstep
        self.pix = pix
        self.show = show_plots
        self.nsteps = nsteps

    def findDiffConstant(self, data):
        """ takes data in the format of 1d tracks , rows: [0-'tag', 1-'t', 2-'mass', 3-'z', 4-'width']

        __future__
        implement possibility of accepting data for multiple particles in one input array
        implement possibility of having tracks with missing datapoints in time
        add option of showing histograms of stepsize distribution
        """
        msd = np.zeros([self.nsteps])
        mean = np.zeros([self.nsteps])
        taxis = np.arange(self.nsteps)
        positions = data[:, 3]
        for n in (np.arange(self.nsteps-1)+1):
            temp = np.roll(positions,n)
            temp = temp - positions
            mean[n] = np.mean(temp[n:])
            msd[n] = np.var(temp[n:])

        pfit = np.polyfit(taxis, msd, 1)
        fit = np.polyval(pfit, taxis)
        if self.show:
            plt.subplot(1, 2, 1)
            plt.scatter(taxis, mean, marker=',')
            plt.subplot(1, 2, 2)
            plt.scatter(taxis, msd, marker='.')
            plt.plot(taxis, fit)
            plt.show()

        print(pfit)
        dif_const = pfit[0] * (self.pix**2) / self.tstep / 2
        return dif_const

    def findDriftZ(self, data):

        return


