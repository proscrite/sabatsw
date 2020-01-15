import pandas as pd
import numpy as np

import glob
import os
import sys

import datetime
import warnings

from ali.ali_sw import load_raw_ali_df
import argparse

def process_dfRaw_peaks(path : str, peakLength : int = 0) -> pd.DataFrame:
    """Method to process ALI raw df into peak df
    Parameters:
    peakLengthCut : int
        Index slice size per peak. If no value is given, computed as minimum
        distance between peaks."""

    def findTroughsFast(dfRaw) -> np.array:
        """Faster trough-finding routine based on dataframe entry 'valve' = on/off"""

        tr0 = np.where(dfRaw['valve'] == 1)[0]
        tr1 = tr0[np.where(tr0[1:] - tr0[:-1] > 1)[0]] # Several consecutive entries may have 'valve' on, pick only the last of each set
        troughs = np.append(tr1, tr0[len(tr0)-1])  # The last peak must be included manually (can never fulfill the previous condition)

        return troughs

    def extractDfPeaks(dfRaw : pd.DataFrame, troughs : np.array, peakLength : int = 0, full : bool = False) -> pd.DataFrame :
        """Transform dfpRaw into matrix and dfPeaks
        Parameters:
        peakLengthCut : int
            Index slice size per peak. If no value is given, computed as minimum
            distance between peaks."""

        if full:  #In this case import whole peak by looping
            m0 = []
            for i, t in enumerate(troughs):
                if i < len(troughs) - 1:
                    m0.append(dfRaw.p_chamber[t : troughs[i+1]].values)

            m0.append(dfRaw.p_chamber[t :].values)
            dfpeak = pd.DataFrame(list(map(np.ravel, m0))).transpose()
            dfpeak.columns = ['peak'+str(i) for i in range(len(troughs))]

        else:
            if peakLength == 0:
                peakLength = int(np.min(troughs[1:] - troughs[:-1]))

            lenP = len(dfRaw.p_chamber)
            troughs = troughs[lenP - troughs > peakLength] # exclude peaks in the end with too short length

            npeaks = troughs.shape[0]
            indices_offset = np.repeat(np.arange(peakLength), npeaks).reshape(peakLength, npeaks)

            dfpeak = pd.DataFrame(dfRaw.p_chamber.values[indices_offset + troughs])
            dfpeak.columns = ['peak'+str(i) for i in range(len(troughs))]

        return dfpeak

    #### Main ####

    dfRaw = load_raw_ali_df(path)
    troughs = findTroughsFast(dfRaw)
    dfpeak = extractDfPeaks(dfRaw, troughs, peakLength = peakLength)

    return dfpeak

def avCurves(dfp : pd.DataFrame):
    """Compute average curve profile, uncertainty area and residue dataframe
    Input:
    -----
    dfp : pd.DataFrame
        Processed peaks from ALI dfRaw"""
    av_p = np.array(dfp.mean(axis=1))
    sd_p = np.array(dfp.std(axis=1))

    dfResidue = av_p.reshape(len(av_p),1) - dfp
    return av_p, sd_p, dfResidue

def plotResiduesDistribution(dfp : pd.DataFrame, nsigma : int = 2) -> list:
    """Plot and fit to gaussian the total residuals (integrated over whole time)
    Return ID of peaks whose residues lay out of the nsigma CI"""
    import matplotlib.pyplot as plt
    _,_, dfResidue = avCurves(dfp)
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

def plotAverageProfile(dfp : pd.DataFrame, flag_p: bool = False, nsigma : int = 2, ax = None, color = 'k', lb : str = ''):
    """Plot average curve profile and the peaks with residuals larger than the specified Confidence Level
    Parameters:
    flag_p: bool
        Flag to include irregular peaks. Default: False
    nsigma: int
        Level of confidence for peaks total residuals. Default: False"""
    import matplotlib.pyplot as plt
    sc = 100 # Timescale: seconds


    av_p, sd_p, _ = avCurves(dfp)

    if ax == None:
        ax = plt.gca()

    if flag_p:
        peaksID = plotResiduesDistribution(dfp, nsigma);
        ax.cla()

        for p in peaksID:
            ax.plot(dfp.index/sc, dfp[p].values, label = p)

    if lb == '': lb = 'Average Profile'
    ax.plot(dfp.index/sc, av_p, color, label = lb)
    ax.fill_between(dfp.index/sc, av_p-sd_p, av_p+sd_p, color=color, alpha=0.1, label = '%i sigma CI' %nsigma)

    ax.legend(loc='upper right')
    ax.set_yscale('log')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pressure [mbar]')
    plt.gcf().set_figwidth(12)
    plt.gcf().set_figheight(12)


def polishDfPeak(dfp, nsigma : int = 2) -> pd.DataFrame:
    """Drop peaks with total residual lying outside of the specified confidence interval
    Returns dfpeak with dropped columns"""

    peaksID = plotResiduesDistribution(nsigma);
    plt.clf();

    dfp.drop(['peak'+str(i) for i in peaksID], axis=1, inplace=True)
    return dfp

def save_processed_peaks(path: str, dfp : pd.DataFrame):
    """Save dfp with .pyk extension to subdirectory 'processed_peaks' of current directory,
    create it if it does not exist"""

    path_array = os.path.split(path)
    path_pyk = path_array[0] + '/processed_peaks/' + path_array[1] + '.pyk'
    try :
        print('Saving processed peaks df to ', path_pyk)
        dfp.to_csv(path_pyk)
    except FileNotFoundError:
        print('Lets try soming else')
        print('processed_peaks subdirectory did not exist yet, creating it...')
        os.mkdir(path_array[0] + '/processed_peaks')

        print('Saving processed peaks df to ', path_pyk)
        dfp.to_csv(path_pyk)
