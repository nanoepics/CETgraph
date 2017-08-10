"""
    CETgraph.tracking.identify.py
    ==================================
    Routines that are necessary for locating particles inside a frame and tracking them in a series of frames or in a waterfall data array

    .. lastedit:: 9/7/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np
from .lib import calc
from scipy import signal

class TracksZ:
    """
    The "TracksZ" module recovers the track coordinates of multiple particles in a waterfall image as they go through
    the 1D field of view. It assumes uni-directional drift towards z = +inf; in __future__ consider bi-directional drift

    parameters needed to initiate the localization requirements
    :param psize: expected particle size (e.g. point spread function)
    :param step: expected step size for searching for next location
    :param snr: expected signal to noise ratio for the peakfinder routine
    :param noiselvl: noise level below which all image data will be set to zero
    :param iniloc: starting position [row, column] of the track passed by the monitoring routine
    :param imgsize: [number of rows, number of columns] in the analyzed image array
    [not implemented] consistently return success/failure messages regarding locating the particle
    """
    def __init__(self, psize=8, step=1, snr=5, noiselvl=1):
        self.noise = noiselvl
        self.psize = psize
        self.snr = snr
        self.step = step
        self.fov = 0
        self.nframes = 0

    def locateInitialPosition(self, data):
        """
         finds initial position peaks in the first few lines; iterates as long as some peaks are found
         :param:
         data: waterfall data array of shape (fov, nframes)
         :return:
         possible location of particles in he first non-empty frame
        """
        psize = self.psize
        fov, nframes = np.shape(data)
        self.fov = fov
        self.nframes = nframes
        min_signal = self.noise  # any value below min_signal will be set to zero, might cause miscalculation of actual intensity
        widths = np.arange(psize - 3, psize + 3)  # peak-widths that are acceptable for peak finder
        peaks_indx = []
        cur_t = 0
        while (len(peaks_indx) == 0 and cur_t < nframes):  # scanning frame by frame until finds at least one peak
            cur_line = data[:, cur_t]
            cur_line[cur_line < min_signal] = 0
            peaks_indx = signal.find_peaks_cwt(cur_line, widths, min_snr=self.snr) #check help of scipy.signal for documentation
            cur_t += 1
        print('Particles found at', peaks_indx)
        return peaks_indx
    ####done only till hear
    def collectTracks(self, data, iniloc):

        # %% turning peak indices into particle positions and initializing the tracks data structure
        columns = ['tag', 't', 'z', 'mass', 'width']
        tracks = DataFrame([[0, 0, 0, 0, 0]], columns=columns)
        cur_line = rawdata[:, cur_t - 1]
        p_tag = len(peaks_indx)  # largest particle tag for the entries of peak_indx
        for pp in np.arange(p_tag, 0, -1):
            p = peaks_indx[pp - 1]
            if p < psize:
                seg = cur_line[0:p + psize]
                ori = 0
            elif p > (pix - psize - 1):
                seg = cur_line[p - psize:pix - 1]
                ori = p - psize
            else:
                seg = cur_line[p - psize:p + psize]
                ori = p - psize
            peak_attr = centroid(seg, ori)
            p_tag = len(peaks_indx)
            new_row = DataFrame([np.concatenate(([p_tag - pp + 1, cur_t - 1], peak_attr))], columns=columns)
            tracks = tracks.append(new_row, ignore_index=True)
        # %%tracing the peaks and checking for new particle arrivals at x=0
        """
        The following simplification are made for the following code
        - drift is not more than pdrift per frame
        - when tracks overlap, the rest of the track is common for two particle
          until the exit the field of view
        - drift is from lower to higher index in each line  

        To be considered in the future
        - colocalization for particles that come too close
        - discontinuation of tracks for particles that suddenly disappear
        - 
        last updated in v0.2, 21 September 2015
        """
        pdrift = 1  # estimated particle drift per frame in pixels
        for t in np.arange(cur_t, frames):
            cur_line = rawdata[:, t]
            for pp in np.arange(len(peaks_indx), 0, -1):
                p = peaks_indx[pp - 1]
                if p < psize:
                    seg = cur_line[0:p + pdrift]
                    p = np.argmax(seg)
                    if p < psize:
                        seg = cur_line[0:p + psize]
                        ori = 0
                    else:
                        seg = cur_line[p - psize:p + psize]
                        ori = p - psize
                    peak_attr = centroid(seg, ori)
                    new_row = DataFrame([np.concatenate(([p_tag - pp + 1, t], peak_attr))], columns=columns)
                    temp = tracks.append(new_row, ignore_index=True)
                    del tracks  # This is to free up memory which gets stuff by pandas append command
                    tracks = temp
                    del temp
                    peaks_indx[pp - 1] = p
                elif p > (pix - pdrift - 1):
                    peaks_indx = np.delete(peaks_indx, pp - 1)  # terminates the track if it is too close to the edge
                else:
                    seg = cur_line[p - psize:p + psize + pdrift]
                    p = p - psize + np.argmax(seg)
                    if p > (pix - psize - 1):
                        peaks_indx = np.delete(peaks_indx, pp - 1)
                    else:
                        seg = cur_line[p - psize:p + psize]
                        ori = p - psize
                    peak_attr = centroid(seg, ori)
                    new_row = DataFrame([np.concatenate(([p_tag - pp + 1, t], peak_attr))], columns=columns)
                    tracks = tracks.append(new_row, ignore_index=True)
                    peaks_indx[pp - 1] = p
            if (len(peaks_indx) == 0 or min(peaks_indx) > pdrift + psize):
                seg = cur_line[0:pdrift + psize]
                if (max(seg) > (2 * min_signal)):
                    newpeak = np.array([np.argmax(seg)])
                    peaks_indx = np.concatenate((newpeak, peaks_indx))
                    p_tag += 1
            if (np.mod(t, 1000) == 0):
                print(t, 'frames analyzed.')
