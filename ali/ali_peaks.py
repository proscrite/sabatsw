import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import glob
import os
import peakutils
import datetime
import warnings
from mpl_toolkits.mplot3d import Axes3D
from ali.ali_sw import *
from dataclasses import dataclass
@dataclass

class ExtractAliPeaks:
    """Method class process ALI raw df into peak df"""

    def __init__(self, path : str):
        self.dfRaw = load_raw_ali_df(path)

    def findTroughsFast(self) -> np.array:
        """Faster trough-finding routine based on dataframe entry 'valve' = on/off"""

        tr0 = np.where(self.dfRaw['valve'] == 1)[0]
        tr1 = tr0[np.where(tr0[1:] - tr0[:-1] > 1)[0]] # Several consecutive entries may have 'valve' on, pick only the last of each set
        troughs = np.append(tr1, tr0[len(tr0)-1])  # The last peak must be included manually (can never fulfill the previous condition)

        return troughs

    def extractDfPeaks(self, peakLength : int = 0, full : bool = False) -> pd.DataFrame :
        """Transform self.dfRaw into matrix and dfPeaks
        Parameters:
        peakLengthCut : int
            Index slice size per peak. If no value is given, computed as minimum
            distance between peaks."""

        troughs = self.findTroughsFast()

        if full:  #In this case import whole peak by looping
            m0 = []
            for i, t in enumerate(troughs):
                if i < len(troughs) - 1:
                    m0.append(self.dfRaw.p_chamber[t : troughs[i+1]].values)

            m0.append(self.dfRaw.p_chamber[t :].values)
            dfpeak = pd.DataFrame(list(map(np.ravel, m0))).transpose()
            dfpeak.columns = ['peak'+str(i) for i in range(len(troughs))]

        else:
            if peakLength == 0:
                peakLength = int(np.min(troughs[1:] - troughs[:-1]))

            lenP = len(self.dfRaw.p_chamber)
            troughs = troughs[lenP - troughs > peakLength] # exclude peaks in the end with too short length

            npeaks = troughs.shape[0]
            indices_offset = np.repeat(np.arange(peakLength), npeaks).reshape(peakLength, npeaks)

            dfpeak = pd.DataFrame(self.dfRaw.p_chamber.values[indices_offset + troughs])
            dfpeak.columns = ['peak'+str(i) for i in range(len(troughs))]

        return dfpeak

class AliPeaks(ExtractAliPeaks):

    def __init__(self, peakLength : int):
        self.df = ExtractAliPeaks.extractDfPeaks(peakLength)

    def avCurves(self):
        """Compute average curve profile, uncertainty area and residue dataframe"""
        av_p = np.array(self.df.mean(axis=1))
        sd_p = np.array(self.df.std(axis=1))

        dfResidue = av_p.reshape(len(av_p),1) - self.df
        return av_p, sd_p, dfResidue

    def plotResiduesDistribution(self, nsigma : int = 2) -> list:
        """Plot and fit to gaussian the total residuals (integrated over whole time)
        Return ID of peaks whose residues lay out of the nsigma CI"""
        _,_, dfResidue = self.avCurves()
        res_peak = dfResidue.sum(axis=0)

        plt.hist(res_peak, bins=10, zorder=-1)

        n, bins = np.histogram(res_peak, bins=10)
        cbins = 0.5*(bins[1:]+bins[:-1])

        from scipy.stats import norm
        (mu, sigma) = norm.fit(res_peak)
        mu, sigma
        xpl = np.linspace(min(cbins), max(cbins), 100)
        ypl = norm.pdf(xpl, mu, sigma)

        xci = np.linspace(mu- nsigma * sigma, mu + nsigma * sigma, 100)
        yci = norm.pdf(xci, mu, sigma)
#         ypl *= n.sum()
#         yci *= n.sum()

        plt.plot(xpl, ypl, 'r', label='Gaussian fit \nmean = %.2e \nsigma = %.2e' %(mu, sigma))
        plt.fill_between(xci, 0, yci, alpha=0.9, color='g', zorder = 1, label='2$\sigma$ CI')
        plt.xlabel('Peak total residuals')
        plt.legend()

        peaksID = np.where(dfResidue.sum(axis=0) < mu - nsigma* sigma)[0]

        return ['peak'+str(i) for i in peaksID]

    def plotAverageProfile(self, flag_p: bool = False, nsigma : int = 2, ax = None,
                           color = 'k', lb : str = ''):
        """Plot average curve profile and the peaks with residuals larger than the specified Confidence Level
        Parameters:
        flag_p: bool
            Flag to include irregular peaks. Default: False
        nsigma: int
            Level of confidence for peaks total residuals. Default: False"""
        plt.figure(figsize=(10,10))
        sc = 100 # Timescale: seconds


        av_p, sd_p, _ = self.avCurves()

        if ax == None:
            ax = plt.gca()

        if flag_p:
            peaksID = self.plotResiduesDistribution(nsigma);
            ax.cla()

            for p in peaksID:
                ax.plot(self.df.index/sc, self.df[p].values, label = p)

        if lb == '': lb = 'Average Profile'
        ax.plot(self.df.index/sc, av_p, color, label = lb)
        ax.fill_between(self.df.index/sc, av_p-sd_p, av_p+sd_p, color=color, alpha=0.1, label = '%i sigma CI' %nsigma)

        ax.legend(loc='upper right')
        ax.set_yscale('log')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Pressure [mbar]')

    def polishDfPeak(self, nsigma : int = 2) -> pd.DataFrame:
        """Drop peaks with total residual lying outside of the specified confidence interval
        Returns dfpeak with dropped columns"""

        peaksID = self.plotResiduesDistribution(nsigma);
        plt.clf();

        self.df.drop(['peak'+str(i) for i in peaksID], axis=1, inplace=True)
        return self.df
