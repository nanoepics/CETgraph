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
    :param step: expected diffusion distance in pixels to search for location in the next frame
    :param drift: expected drift in pixels per frame
    :param snr: expected signal to noise ratio for the peakfinder routine
    :param noiselvl: noise level below which all image data will be set to zero
    :param locations: location of particles in the last processed frame

    [not implemented] consistently return success/failure messages regarding locating the particle
    """
    def __init__(self, psize=8, step=1, drift=1, snr=5, noiselvl=1):
        self.noise = noiselvl
        self.psize = psize
        self.snr = snr
        self.step = step
        self.fov = 0
        self.nframes = 0
        self.drift = int(drift)
        self.fnumber = 0
        self.locations = []

    def locateInitialPosition(self, data):
        """
         finds initial position peaks in the first few lines; iterates as long as some peaks are found
         :param:
         data: waterfall data array with shape:(fov, nframes)
         :returns:
         possible location of particles in he first non-empty frame
        """
        psize = self.psize
        fov, nframes = np.shape(data)
        self.fov = fov
        self.nframes = nframes
        min_signal = self.noise  # any value below min_signal will be set to zero, might cause miscalculation of actual intensity
        widths = np.arange(psize - 4, psize + 6)  # peak-widths that are acceptable for peak finder
        peak_indices = []
        cur_t = 0
        while (len(peak_indices) == 0 and cur_t < nframes):  # scanning frame by frame until finds at least one peak
            cur_line = data[:, cur_t]
            cur_line[cur_line < min_signal] = 0
            peak_indices = np.array([np.argmax(cur_line)])
            #peak_indices = signal.find_peaks_cwt(cur_line, widths, min_snr=self.snr) #check help of scipy.signal for documentation
            cur_t += 1

        self.locations = peak_indices
        self.fnumber = cur_t - 1
        if not peak_indices:
            print('No peaks found! Try different parameters.')
        else:
            print('Particles found in frame %d at positions' %(cur_t -1), peak_indices)
        return peak_indices

    def collectTracks(self, data, loca = []):
        """

        :param data: waterfall data array with shape:(fov, nframes)
        :param loca: initial position of peaks in the first frame, if set to empty, method will try to find initial location
        :param vdrift: estimated particle drift per frame
        :return: array with all particle coordinates in the format [0-'tag', 1-'t', 2-'mass', 3-'z', 4-'width']
        not that time is explicitly mentioned for the __future__ cases that particle might be missing for a few frame
        """
        psize = self.psize
        step = self.step
        drift = self.drift
        fov, nframes = np.shape(data)
        self.fov = fov
        self.nframes = nframes
        if not loca:
            particles = self.locateInitialPosition(data)
            cur_t = self.fnumber
        else:
            particles = loca
            cur_t = 0
        tracks = []
        cur_line = data[:, cur_t]
        p_count = len(particles)  # number of particles give largest particle tag for the entries of peak_indx
        print(p_count)
        for pp in np.arange(len(particles), 0, -1):
            p = particles[pp - 1]
            if p < psize:
                seg = cur_line[0:p + psize]
                ori = 0
            elif p > (fov - psize - 1):
                seg = cur_line[p - psize:fov]
                ori = p - psize
            else:
                seg = cur_line[p - psize:p + psize]
                ori = p - psize
            peak_attr = calc.centroid1D(seg, ori)
            p_tag = p_count - pp + 1
            if not len(tracks):
                tracks = np.array([np.concatenate(([p_tag, cur_t], peak_attr))])
            else:
                new_row = np.concatenate(([p_tag, cur_t], peak_attr))
                tracks = np.concatenate((tracks, [new_row]), axis=0)
        cur_t += 1
        # %%
        """
        Next comes tracing the peaks and checking for new particle arrivals at x=0
        The following simplification are made for the following code
        - drift is not more than pdrift per frame
        - when tracks overlap, the rest of the track is common for two particle
          until the exit the field of view
        - drift is from lower to higher index in each line  

        To be considered in the future
        - colocalization for particles that come too close
        - discontinuation of tracks for particles that suddenly disappear

        """
        for t in np.arange(cur_t, nframes):
            self.fnumber = t
            cur_line = data[:, t]
            for pp in np.arange(len(particles), 0, -1):
                p = particles[pp - 1]
                if p < psize:
                    seg = cur_line[0:p + drift]
                    p = np.argmax(seg)
                    if p < psize:
                        seg = cur_line[0:p + psize]
                        ori = 0
                    else:
                        seg = cur_line[p - psize:p + psize]
                        ori = p - psize
                    peak_attr = calc.centroid1D(seg, ori)
                    p_tag = p_count - pp + 1
                    new_row = np.concatenate(([p_tag, t], peak_attr))
                    tracks = np.concatenate((tracks, [new_row]), axis=0)
                    particles[pp - 1] = int(peak_attr[1])
                elif p > (fov - drift - 1):
                    particles = np.delete(particles, pp - 1)  # terminates the track if it is too close to the edge
                else:
                    seg = cur_line[p - psize:p + psize + drift]
                    ori = p - psize + np.argmax(seg)
                    # if p > (fov - psize - 1):
                    #     particles = np.delete(particles, pp - 1)
                    # else:
                    #     seg = cur_line[p - psize:p + psize]
                    #     ori = p - psize
                    peak_attr = calc.centroid1D(seg, ori)
                    p_tag = p_count - pp + 1
                    new_row = np.concatenate(([p_tag, t], peak_attr))
                    tracks = np.concatenate((tracks, [new_row]), axis=0)
                    particles[pp - 1] = int(peak_attr[1])
            if (len(particles) == 0 or min(particles) > drift + psize):  #checking if there is no particle in the left entry of the frame
                seg = cur_line[0:drift + psize]
                if (max(seg) > (self.snr * self.noise)):
                    newpeak = np.array([np.argmax(seg)])
                    particles = np.concatenate((newpeak, particles))
                    p_count += 1
            if (np.mod(t, 1000) == 0):
                print(t+1, 'frames analyzed.')

        return tracks
